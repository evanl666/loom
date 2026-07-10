"""External artifact store: git-lfs for oversized tool results.

Browser screenshots, DOM snapshots, big SQL result sets, support attachments --
a generic debugger will hit results too large to sit inline in a trace. This
externalizes any tool result over a threshold to a content-addressed blob dir,
leaving a small pointer in the trace:

    loom artifacts externalize run.loom.json --threshold 32kb   # trace shrinks
    loom artifacts inline run.loom.json                         # restore in full

The pointer records the sha, byte size, and whether the blob was scrubbed, so a
shrunk trace still Studio-renders ("📎 externalized artifact, 45 KB") and
`loom pack`/`serve` can resolve or ship the blobs. Externalize for
storage/transport; inline before replay (a pointer is not the content).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

_MARK = "_loom_artifact"


def _is_sha256(s: str) -> bool:
    """A blob name is exactly a 64-char lowercase hex digest -- nothing else is
    allowed near os.path.join, so an untrusted pointer can't traverse out of the
    blob dir (../.., an absolute path, /dev/zero, a fifo...)."""
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _blobdir_for(trace_path: str, blobdir: str) -> str:
    return blobdir or (trace_path[: -len(".loom.json")] if trace_path.endswith(".loom.json")
                       else trace_path) + ".artifacts"


def _is_pointer(value) -> bool:
    return isinstance(value, dict) and _MARK in value


def externalize(data: dict, blobdir: str, threshold: int = 32_768) -> "tuple[dict, list[dict]]":
    """Move oversized string tool results to ``blobdir``; return (data, manifest).

    Only tool-result effects are externalized (model effects carry the
    replay-critical wire data). The trace dict is copied, not mutated.
    """
    os.makedirs(blobdir, exist_ok=True)
    out = copy.deepcopy(data)
    manifest: list[dict] = []
    scrubbed = bool(data.get("scrubbed"))
    for e in out.get("log", []):
        if not e.get("kind", "").startswith("tool:"):
            continue
        result = e.get("result")
        if _is_pointer(result) or not isinstance(result, str):
            continue
        blob = result.encode()
        if len(blob) < threshold:
            continue
        sha = hashlib.sha256(blob).hexdigest()
        with open(os.path.join(blobdir, sha), "wb") as f:
            f.write(blob)
        pointer = {"sha": sha, "bytes": len(blob), "kind": e["kind"],
                   "scrubbed": scrubbed}
        e["result"] = {_MARK: pointer}
        manifest.append({"seq": e.get("seq"), **pointer})
    return out, manifest


def inline(data: dict, blobdir: str) -> "tuple[dict, list[str]]":
    """Restore externalized results from ``blobdir``; return (data, missing shas)."""
    out = copy.deepcopy(data)
    missing: list[str] = []
    for e in out.get("log", []):
        result = e.get("result")
        if not _is_pointer(result):
            continue
        sha = result[_MARK].get("sha", "")
        if not _is_sha256(sha):
            missing.append(sha)  # malformed/hostile pointer: never touch the path
            continue
        path = os.path.join(blobdir, sha)
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except OSError:
            missing.append(sha)
            continue
        if hashlib.sha256(blob).hexdigest() != sha:
            missing.append(sha)  # tampered / wrong blob
            continue
        e["result"] = blob.decode(errors="replace")
    return out, missing


def artifact_pointer(value) -> "dict | None":
    """The pointer metadata if ``value`` is an externalized artifact, else None."""
    return value[_MARK] if _is_pointer(value) else None


def preview(pointer: dict) -> str:
    kb = pointer.get("bytes", 0) / 1024
    tag = " (scrubbed)" if pointer.get("scrubbed") else ""
    return f"📎 externalized artifact, {kb:.0f} KB{tag} — sha {pointer.get('sha', '')[:10]}…"
