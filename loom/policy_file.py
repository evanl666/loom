"""Policy-as-code: named safety profiles and ``loom-policy.yml``.

Writing a wall of ``--deny``/``--confirm`` globs on every run doesn't scale.
A policy is a named, versionable document instead -- ship a built-in profile
or point at your own file:

    loom record claude "fix the test" --profile claude-code-safe
    loom record claude "fix the test" --policy loom-policy.yml

A policy resolves to the same Shield the flags build, so everything the
firewall can do (deny/allow/confirm precedence, sequence tripwires, the
default action) is expressible. Files are YAML (or JSON); YAML is read with
a tiny bounded parser so the zero-dependency install still works, and with
PyYAML if it happens to be installed.
"""

from __future__ import annotations

import json

# Built-in profiles. Each is exactly the keyword arguments Shield takes, so a
# profile is just a policy that ships with loom. Kept deliberately small and
# readable -- they are the thing users copy and adapt.
PROFILES: "dict[str, dict]" = {
    "claude-code-safe": {
        "description": "Sane defaults for a coding agent run with permissions skipped: "
                       "reads and test runs flow, network/installs/pushes ask, secrets "
                       "and destructive shell are blocked, egress after a secret read is cut.",
        "default": "confirm",
        "allow": [
            "Read(*)", "Glob(*)", "Grep(*)", "LS(*)", "WebSearch(*)",
            "Bash(*pytest*)", "Bash(*npm test*)", "Bash(*npm run test*)",
            "Bash(*go test*)", "Bash(*cargo test*)", "Bash(*ls *)", "Bash(*git status*)",
            "Bash(*git diff*)", "Edit(*)", "Write(*)",
        ],
        "confirm": [
            "Bash(*curl*)", "Bash(*wget*)", "Bash(*git push*)",
            "Bash(*npm install*)", "Bash(*pip install*)", "WebFetch(*)",
        ],
        "deny": [
            "Read(*.env*)", "Read(*/.ssh/*)", "Read(*/.aws/*)", "Read(*secrets*)",
            "Bash(*rm -rf*)", "Bash(*curl* | *sh*)", "Bash(*:(){*", "Bash(*mkfs*)",
        ],
        "sequence": [
            "after Read(*.env*): deny WebFetch*, deny Bash(*curl*), deny Bash(*wget*)",
            "taint sk-ant-*: confirm *",
            "taint sk-proj-*: confirm *",
        ],
    },
    "ci-safe": {
        "description": "Non-interactive: nothing waits for a human (no confirms), "
                       "read-only plus tests allowed, everything else denied.",
        "default": "deny",
        "allow": [
            "Read(*)", "Glob(*)", "Grep(*)", "LS(*)",
            "Bash(*pytest*)", "Bash(*npm test*)", "Bash(*go test*)",
        ],
        "deny": ["Read(*.env*)", "Read(*/.ssh/*)"],
        "sequence": ["after Read(*.env*): deny *"],
    },
    "prod-data-safe": {
        "description": "For agents near real data: reads ask, any write/delete/egress "
                       "is denied, a credential sighting locks everything to confirm.",
        "default": "confirm",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "WebSearch(*)"],
        "deny": [
            "Bash(*rm*)", "Bash(*DROP *)", "Bash(*DELETE *)", "Bash(*curl*)",
            "Bash(*wget*)", "Write(*)", "Edit(*)", "Read(*.env*)", "Read(*/.ssh/*)",
        ],
        "sequence": ["taint sk-*: confirm *", "taint AKIA*: confirm *"],
    },
    "prod-db-safe": {
        "description": "Database work against production: SELECTs flow, anything that "
                       "mutates schema or rows asks, drops/truncates/deletes are blocked.",
        "default": "confirm",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "Bash(*SELECT *)", "Bash(*EXPLAIN *)",
                  "Bash(*\\\\d*)", "Bash(*SHOW *)"],
        "confirm": ["Bash(*INSERT *)", "Bash(*UPDATE *)", "Bash(*ALTER *)",
                    "Bash(*CREATE *)", "Bash(*migrate*)"],
        "deny": ["Bash(*DROP *)", "Bash(*TRUNCATE *)", "Bash(*DELETE FROM*)",
                 "Read(*.env*)", "cap:destructive"],
        "sequence": ["taint password*: confirm *", "taint sk-*: confirm *"],
    },
    "github-actions-safe": {
        "description": "Running inside CI: fully non-interactive (nothing waits for a "
                       "human), read/build/test allowed, secrets and egress denied.",
        "default": "deny",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "LS(*)", "Write(*)", "Edit(*)",
                  "Bash(*pytest*)", "Bash(*npm test*)", "Bash(*npm run build*)",
                  "Bash(*go test*)", "Bash(*cargo test*)", "Bash(*make *)",
                  "Bash(*git status*)", "Bash(*git diff*)", "Bash(*git log*)"],
        "deny": ["Read(*.env*)", "Read(*/.ssh/*)", "cap:secret",
                 "Bash(*curl*)", "Bash(*wget*)", "WebFetch(*)"],
        "sequence": ["after Read(*secret*): deny *"],
    },
    "k8s-safe": {
        "description": "Cluster operations: get/describe/logs flow, apply/scale ask, "
                       "delete/drain and anything against kube-system is blocked.",
        "default": "confirm",
        "allow": ["Read(*)", "Glob(*)", "Grep(*)", "Bash(*kubectl get*)",
                  "Bash(*kubectl describe*)", "Bash(*kubectl logs*)",
                  "Bash(*kubectl top*)", "Bash(*helm list*)", "Bash(*helm status*)"],
        "confirm": ["Bash(*kubectl apply*)", "Bash(*kubectl scale*)",
                    "Bash(*kubectl rollout*)", "Bash(*helm upgrade*)", "Bash(*helm install*)"],
        "deny": ["Bash(*kubectl delete*)", "Bash(*kubectl drain*)",
                 "Bash(*kube-system*)", "Bash(*kubectl exec*)", "Read(*/.kube/config*)"],
        "sequence": ["taint token*: confirm *"],
    },
    "customer-data-safe": {
        "description": "Near PII: aggregate reads ask, exports/joins to the outside are "
                       "blocked, any credential or a data sighting cuts egress.",
        "default": "confirm",
        "allow": ["Glob(*)", "Grep(*)"],
        "deny": ["cap:network", "Bash(*COPY *)", "Bash(*\\\\copy*)", "Bash(*mysqldump*)",
                 "Bash(*pg_dump*)", "Write(*export*)", "Read(*.env*)"],
        "sequence": ["taint *@*.*: deny cap:network", "taint sk-*: confirm *"],
    },
}

_SHIELD_KEYS = ("default", "allow", "confirm", "deny", "sequence")


def profile_names() -> "list[str]":
    return sorted(PROFILES)


def to_shield_kwargs(doc: dict) -> dict:
    """Extract the Shield-constructor keys from a policy document."""
    kwargs = {k: doc[k] for k in _SHIELD_KEYS if k in doc}
    for listkey in ("allow", "confirm", "deny", "sequence"):
        kwargs.setdefault(listkey, [])
    return kwargs


def resolve(profile: str = "", policy_path: str = "") -> dict:
    """Resolve a --profile name and/or a --policy file into one document.

    A file may itself select a profile (``profile: claude-code-safe``) and add
    to it; explicit lists in the file extend the profile's lists, and a
    ``default`` in the file overrides. Returns a policy document (dict).
    """
    doc: dict = {}
    if profile:
        if profile not in PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; built-in: {', '.join(profile_names())}"
            )
        doc = _clone(PROFILES[profile])

    if policy_path:
        loaded = load_document(policy_path)
        named = loaded.get("profile")
        if named:
            if named not in PROFILES:
                raise ValueError(f"policy file selects unknown profile {named!r}")
            base = _clone(PROFILES[named])
            doc = _merge(base, doc) if doc else base
        doc = _merge(doc, loaded)
    return doc


def load_document(path: str) -> dict:
    """Load a policy file. A top-level ``profiles:`` map is supported (returns
    the sole entry, or requires the file to also name a ``profile:``)."""
    with open(path) as f:
        text = f.read()
    data = _parse(text, path)
    if "profiles" in data and set(data) <= {"profiles", "profile", "version"}:
        profiles = data["profiles"]
        chosen = data.get("profile")
        if chosen:
            if chosen not in profiles:
                raise ValueError(f"{path}: profile {chosen!r} not in this file")
            return profiles[chosen]
        if len(profiles) == 1:
            return next(iter(profiles.values()))
        raise ValueError(
            f"{path} defines {len(profiles)} profiles; pick one with `profile: <name>` "
            f"at the top level (or --profile on the command line)"
        )
    return data


def lint(doc: dict) -> "list[str]":
    """Catch the misconfigurations that make a policy quietly not work.

    The classic footgun: ``deny: rm -rf`` looks like it blocks ``rm -rf`` but
    actually targets a TOOL NAMED 'rm -rf', which never exists -- so it never
    fires. We flag command-shaped patterns, wildcard-less signatures, rules
    shadowed by a broader deny, and an empty policy.
    """
    warnings: list[str] = []
    kw = to_shield_kwargs(doc)
    all_patterns = [(a, p) for a in ("deny", "allow", "confirm") for p in kw.get(a, [])]
    if not all_patterns and not kw.get("sequence") and doc.get("default", "allow") == "allow":
        warnings.append("policy is empty and defaults to allow -- it blocks nothing")

    for action, p in all_patterns:
        has_sig = "(" in p and p.endswith(")")
        # A pattern with a space but no signature parens targets a tool *named*
        # that string -- almost always a mistake for a Bash command.
        if " " in p and not has_sig:
            cmd = p.strip("*")
            warnings.append(
                f"{action} '{p}': matches a TOOL NAMED '{p}', not a command. "
                f"For a shell command use a signature glob, e.g. `Bash(*{cmd}*)`."
            )
        # A signature with no wildcard only matches that exact argument string.
        elif has_sig:
            inside = p[p.index("(") + 1: -1]
            if inside and "*" not in inside and "?" not in inside:
                warnings.append(
                    f"{action} '{p}': the argument has no wildcard, so it matches only "
                    f"that exact value. Did you mean `{p[:p.index('(')]}(*{inside}*)`?"
                )

    # An allow shadowed by a broader deny never takes effect (deny > allow).
    from fnmatch import fnmatch

    for a in kw.get("allow", []):
        for d in kw.get("deny", []):
            if fnmatch(a, d) or a == d:
                warnings.append(f"allow '{a}' is shadowed by deny '{d}' (deny wins)")
    return warnings


def _clone(d: dict) -> dict:
    return json.loads(json.dumps(d))


def _merge(base: dict, extra: dict) -> dict:
    """Extend base's lists with extra's, override scalars."""
    out = _clone(base)
    for k, v in extra.items():
        if isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = v
    return out


def _parse(text: str, path: str) -> dict:
    try:
        import yaml  # optional; handles the full language when present

        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    if text.lstrip().startswith("{"):
        return json.loads(text)
    return _mini_yaml(text, path)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]  # quoted: taken verbatim, may contain '#' or ':'
    # unquoted: an inline ' # comment' ends the value
    hashpos = s.find(" #")
    if hashpos != -1:
        s = s[:hashpos].rstrip()
    return s


def _mini_yaml(text: str, path: str) -> dict:
    """A bounded YAML reader for the policy schema: nested mappings, lists of
    scalar strings, and scalar values. Not a general YAML parser -- it rejects
    what it doesn't understand rather than guessing.

    A key with an empty value opens a block whose kind (list vs mapping) is
    decided by its first child line; ``_Block`` holds it until then.
    """
    root: dict = {}
    stack: "list[tuple[int, object]]" = [(-1, root)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        content = raw.strip()

        while indent <= stack[-1][0]:
            stack.pop()
        container = _materialize(stack[-1][1])

        if content.startswith("- "):
            if not isinstance(container, (list, _Block)):
                raise ValueError(f"{path}: list item outside a list: {content!r}")
            container.append(_unquote(content[2:]))
        elif ":" in content:
            if not isinstance(container, (dict, _Block)):
                raise ValueError(f"{path}: mapping key inside a list: {content!r}")
            key, _, value = content.partition(":")
            key = key.strip()
            if value.strip() == "":
                block = _Block()
                container[key] = block
                stack.append((indent, block))
            else:
                container[key] = _unquote(value)
        else:
            raise ValueError(f"{path}: cannot parse line: {content!r}")

    return _finalize(root)


class _Block:
    """A key's not-yet-typed child block: becomes a list or a dict on first use."""

    def __init__(self):
        self.value: "list | dict | None" = None

    def append(self, item) -> None:
        if self.value is None:
            self.value = []
        self.value.append(item)

    def __setitem__(self, k, v) -> None:
        if self.value is None:
            self.value = {}
        self.value[k] = v


def _materialize(node):
    return node.value if isinstance(node, _Block) and node.value is not None else node


def _finalize(node):
    if isinstance(node, _Block):
        node = node.value if node.value is not None else {}
    if isinstance(node, dict):
        return {k: _finalize(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_finalize(v) for v in node]
    return node
