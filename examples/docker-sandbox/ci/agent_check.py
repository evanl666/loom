"""Runs INSIDE the jail. Exit 0 iff the jail jails and the door works."""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ["ANTHROPIC_BASE_URL"]


def post(url):
    req = urllib.request.Request(
        url,
        data=json.dumps({"model": "m", "max_tokens": 64,
                         "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"content-type": "application/json", "x-api-key": "jailed"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


# 1. Direct egress must fail: example.com, and the upstream fake-api by name
#    (it lives on the egress network -- the jail can't even resolve it).
for url in ("https://example.com", "http://fake-api:9000/v1/messages"):
    try:
        urllib.request.urlopen(url, timeout=5)
        print(f"JAILBREAK: reached {url} directly", flush=True)
        sys.exit(1)
    except (urllib.error.URLError, OSError):
        print(f"blocked as expected: {url}", flush=True)

# 2. The sanctioned door must work (retry while the proxy pip-installs).
# An HTTP status is an ANSWER, not "not up yet" -- fail fast with the body.
deadline = time.time() + 120
while True:
    try:
        got = post(BASE + "/v1/messages")
        break
    except urllib.error.HTTPError as e:
        print(f"proxy answered {e.code}: {e.read()[:300]!r}", flush=True)
        sys.exit(1)
    except (urllib.error.URLError, OSError) as e:
        if time.time() > deadline:
            print(f"proxy never came up: {e}", flush=True)
            sys.exit(1)
        print(f"waiting for proxy: {e}", flush=True)
        time.sleep(2)

text = got["content"][0]["text"]
assert text == "recorded through the only door", text
print("through the proxy:", text, flush=True)
print("JAIL-OK", flush=True)
