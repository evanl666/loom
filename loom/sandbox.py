"""Sandbox binding: make the recording proxy the agent's ONLY network door.

The proxy alone is a camera on a door in an open field -- an agent (or its
model) can simply make other connections and walk around it. Combined with an
OS sandbox that denies all network except the proxy port, the picture changes:
every model exchange MUST pass through the recorder, so Shield rules and
sequence tripwires cannot be bypassed, and the trace really is the complete
account of what the model saw and said.

    loom record --sandbox --deny 'Read(*.env*)' -- python my_agent.py

macOS: implemented with ``sandbox-exec`` (the same mechanism Claude Code's own
sandboxing uses). Linux/Windows: not built in yet -- run the agent in a
container with no egress and publish only the proxy port; the README shows a
docker recipe.

This confines the network, not the filesystem: the agent can still read and
write files as your user. For filesystem confinement too, ``--container`` runs
the agent in Docker (repo mounted, API routed to the host proxy):

    loom record --container claude "fix the tests" --safe

"Shield controls what the model may ask for; the sandbox controls what the
process can actually do."
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

_PROFILE = """\
(version 1)
(allow default)
(deny network*)
(allow network-outbound (remote unix))
{rules}
"""


def sandbox_profile(ports: "list[int]", allow: "list[str]" = ()) -> str:
    """A sandbox-exec profile: no network except the given loopback ports.

    ``allow`` adds extra ``host:port`` escape hatches (e.g. a local package
    index) -- each one widens the hole, so they are opt-in and explicit.
    """
    rules = [f'(allow network* (remote tcp "localhost:{p}"))' for p in ports]
    rules += [f'(allow network* (remote tcp "{spec}"))' for spec in allow]
    return _PROFILE.format(rules="\n".join(rules))


def wrap_sandboxed(command: "list[str]", ports: "list[int]",
                   allow: "list[str]" = ()) -> "tuple[list[str], str]":
    """Wrap a command so it runs with network access ONLY to ``ports``.

    Returns (wrapped command, profile path) -- unlink the profile when the
    child exits. Raises ``RuntimeError`` on platforms without a built-in
    sandbox.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "--sandbox is built in on macOS only (sandbox-exec). On Linux, run the "
            "agent in a container with no egress and ANTHROPIC_BASE_URL pointed at "
            "the proxy -- see the README's sandbox section for a docker recipe."
        )
    fd, path = tempfile.mkstemp(suffix=".sb", prefix="loom-sandbox-")
    with os.fdopen(fd, "w") as f:
        f.write(sandbox_profile(ports, allow))
    return ["sandbox-exec", "-f", path, *command], path


def wrap_container(command: "list[str]", port: int, image: str = "python:3.12-slim",
                   workdir: "str | None" = None, target: str = "",
                   read_only: bool = False, memory: str = "", cpus: str = "") -> "list[str]":
    """Run the agent command inside a Docker container.

    Shield controls what the model may *ask* for; a container controls what the
    process can actually *do*. The repo is mounted at /workspace, the API is
    routed to the host proxy via ``host.docker.internal`` (so it's still
    recorded and firewalled), and the container is torn down after.

    ``read_only`` mounts the repo read-only with a writable overlay for edits;
    the default mounts it read-write so the workspace diff is captured on the
    host. Raises ``RuntimeError`` if Docker isn't available.
    """
    if not shutil.which("docker"):
        raise RuntimeError(
            "--container needs Docker on PATH. Install Docker, or use --sandbox "
            "(macOS, network-only), or the examples/docker-sandbox compose topology."
        )
    workdir = workdir or os.getcwd()
    env_var = "OPENAI_BASE_URL" if "openai" in target else "ANTHROPIC_BASE_URL"
    base = f"http://host.docker.internal:{port}" + ("/v1" if "openai" in target else "")
    mount = f"{workdir}:/workspace" + (":ro" if read_only else "")
    limits = []
    if memory:
        limits += ["--memory", memory]
    if cpus:
        limits += ["--cpus", cpus]
    return [
        "docker", "run", "--rm", "--init",
        "-v", mount, "-w", "/workspace",
        # Reach the host proxy (Docker Desktop provides this; Linux needs the
        # host-gateway mapping, added explicitly).
        "--add-host", "host.docker.internal:host-gateway",
        "-e", f"{env_var}={base}",
        *limits,
        image, *command,
    ]
