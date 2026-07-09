"""Tools: plain Python functions the agent can call.

Decorate a function with ``@tool`` and its name, docstring, and type hints become
the schema the model sees. No base classes, no registries -- just functions.

    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


@dataclass
class Tool:
    """A callable tool with a JSON-schema view for the model."""

    name: str
    description: str
    fn: Callable[..., Any]
    input_schema: dict
    # Optional capability contract (read/write/exec/network/secret/destructive/
    # idempotent). When set, policy `cap:` rules trust this over inference.
    capabilities: "set[str] | None" = None

    def schema(self) -> dict:
        """The neutral schema passed to a provider."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def _build_schema(fn: Callable) -> dict:
    """Derive a JSON Schema for a function's parameters from its type hints."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    props: dict = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        hint = hints.get(pname, str)
        json_type = _PY_TO_JSON.get(hint, "string")
        props[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def tool(
    fn: "Callable | None" = None,
    *,
    name: "str | None" = None,
    description: "str | None" = None,
    capabilities: "set[str] | None" = None,
) -> Any:
    """Turn a function into a ``Tool``. Usable bare (``@tool``) or with args.

    ``capabilities`` declares the tool's capability contract (read/write/exec/
    network/secret/destructive/idempotent) so ``cap:`` policy rules can trust
    it instead of inferring from the name.
    """

    def wrap(f: Callable) -> Tool:
        return Tool(
            name=name or f.__name__,
            description=(description or inspect.getdoc(f) or "").strip(),
            fn=f,
            input_schema=_build_schema(f),
            capabilities=set(capabilities) if capabilities else None,
        )

    return wrap(fn) if fn is not None else wrap
