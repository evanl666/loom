#!/usr/bin/env bash
# The loom flight-recorder demo -- five acts, fully offline, no API key.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf artifacts && mkdir artifacts

act() { printf '\n\033[1;36m== Act %s: %s ==\033[0m\n\n' "$1" "$2"; }

act 1 "the deploy bot fails -- and every step is recorded"
python agent.py artifacts/flight.loom.json

act 2 "read the flight recording (offline, zero API calls)"
loom replay artifacts/flight.loom.json
echo
loom timeline artifacts/flight.loom.json
echo
loom doctor artifacts/flight.loom.json || echo "(doctor exits 1: findings above)"
loom export artifacts/flight.loom.json -o artifacts/flight.html

act 3 "heal -- find and verify the context repair, keep it as a test"
loom heal artifacts/flight.loom.json --agent agent:build_agent \
    --forbid FAILED --save-regression artifacts/regressions

act 4 "agent CI -- would this one-line prompt change re-break it?"
loom impact artifacts/regressions --agent agent:build_agent
echo
if loom impact artifacts/regressions --agent agent_v2:build_agent; then
    echo "ERROR: impact should have flagged the change" >&2; exit 1
else
    echo "(impact exits 1: the 'innocent' prompt PR changes this recorded run)"
fi

act 5 "the firewall -- same recorder, now with rules"
python fake_model.py --port 8977 & FAKE=$!
trap 'kill $FAKE 2>/dev/null || true' EXIT
sleep 0.7
loom record --save artifacts/sneaky.loom.json --target http://127.0.0.1:8977 \
    --deny 'read_file(*.env*)' -- python sneaky_client.py

printf '\n\033[1mAll five acts ran offline.\033[0m\n'
echo "open the recording:  loom studio artifacts/flight.loom.json"
echo "search the corpus:   loom search artifacts 'shield:deny'"
