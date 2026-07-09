"""Structured output: parse and validate the final answer at the Effect boundary.

Give ``Agent(output_type=...)`` a dataclass, TypedDict, or pydantic model. The
schema is appended to the system prompt, and when the model produces its final
answer the harness parses it -- a failed parse feeds the error back as context
and retries, and each retry is an ordinary recorded model effect, so validated
runs replay deterministically like everything else.

Zero dependencies: dataclasses and TypedDicts are validated natively; pydantic
models are used when pydantic is installed (never required).
"""

from __future__ import annotations

import dataclasses
import json
import typing
from typing import Any

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class OutputInvalid(ValueError):
    """The model's final answer did not match the requested output_type."""


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")


def _is_pydantic(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "model_validate")


def schema_for(tp: Any) -> dict:
    """A JSON schema describing ``tp`` (dataclass / TypedDict / pydantic / builtin)."""
    if _is_pydantic(tp):
        return tp.model_json_schema()
    if tp in _JSON_TYPES:
        return {"type": _JSON_TYPES[tp]}
    origin = typing.get_origin(tp)
    if origin in (list, tuple):
        args = typing.get_args(tp)
        item = schema_for(args[0]) if args else {}
        return {"type": "array", "items": item}
    if origin is dict:
        args = typing.get_args(tp)
        value = schema_for(args[1]) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": value}
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:  # Optional[X]
            return schema_for(args[0])
        return {"anyOf": [schema_for(a) for a in args]}
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        props = {f.name: schema_for(hints[f.name]) for f in dataclasses.fields(tp)}
        required = [
            f.name
            for f in dataclasses.fields(tp)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        return {"type": "object", "properties": props, "required": required}
    if _is_typeddict(tp):
        hints = typing.get_type_hints(tp)
        required_keys = getattr(tp, "__required_keys__", frozenset(hints))
        return {
            "type": "object",
            "properties": {k: schema_for(v) for k, v in hints.items()},
            "required": sorted(required_keys),
        }
    return {}  # Any / unknown: accept whatever parses


def format_instruction(tp: Any) -> str:
    """The system-prompt suffix that asks for JSON matching ``tp``."""
    return (
        "When you give your final answer, respond with ONLY a JSON object "
        "(no prose, no code fences) matching this schema:\n"
        + json.dumps(schema_for(tp), indent=2)
    )


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of ``text``, tolerant of fences/prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    # Every position where a JSON value could START, in order. Prose can contain
    # a stray '{' ("the object { should...") BEFORE the real JSON, so try each
    # candidate until one actually decodes rather than only the first.
    starts = sorted(i for i, c in enumerate(stripped) if c in "{[")
    last_error = None
    for i in starts:
        try:
            value, _ = decoder.raw_decode(stripped[i:])
            return value
        except json.JSONDecodeError as e:
            last_error = e
    if not starts:
        try:  # a bare JSON scalar ("null", a number, a quoted string)
            value, _ = decoder.raw_decode(stripped)
            return value
        except json.JSONDecodeError:
            raise OutputInvalid("no JSON object found in the answer") from None
    raise OutputInvalid(f"invalid JSON: {last_error}") from None


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Validate ``value`` against ``tp``, returning the typed result."""
    if tp is Any or tp is None:
        return value
    if _is_pydantic(tp):
        try:
            return tp.model_validate(value)
        except Exception as e:
            raise OutputInvalid(f"{path}: {e}") from None
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path)
            except OutputInvalid as e:
                errors.append(str(e))
        raise OutputInvalid(f"{path}: matched no option ({'; '.join(errors)})")
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise OutputInvalid(f"{path}: expected an array, got {type(value).__name__}")
        args = typing.get_args(tp)
        item_tp = args[0] if args else Any
        return [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise OutputInvalid(f"{path}: expected an object, got {type(value).__name__}")
        args = typing.get_args(tp)
        value_tp = args[1] if len(args) == 2 else Any
        return {k: _coerce(v, value_tp, f"{path}.{k}") for k, v in value.items()}
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise OutputInvalid(f"{path}: expected an object, got {type(value).__name__}")
        hints = typing.get_type_hints(tp)
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name in value:
                kwargs[f.name] = _coerce(value[f.name], hints[f.name], f"{path}.{f.name}")
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise OutputInvalid(f"{path}: missing required field {f.name!r}")
        return tp(**kwargs)
    if _is_typeddict(tp):
        if not isinstance(value, dict):
            raise OutputInvalid(f"{path}: expected an object, got {type(value).__name__}")
        hints = typing.get_type_hints(tp)
        required_keys = getattr(tp, "__required_keys__", frozenset(hints))
        for k in required_keys:
            if k not in value:
                raise OutputInvalid(f"{path}: missing required key {k!r}")
        return {
            k: _coerce(v, hints[k], f"{path}.{k}") if k in hints else v
            for k, v in value.items()
        }
    if tp is bool:
        if not isinstance(value, bool):
            raise OutputInvalid(f"{path}: expected a boolean, got {type(value).__name__}")
        return value
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in _JSON_TYPES:
        if not isinstance(value, tp) or isinstance(value, bool) and tp is not bool:
            raise OutputInvalid(
                f"{path}: expected {_JSON_TYPES[tp]}, got {type(value).__name__}"
            )
        return value
    return value  # unknown annotation: pass through


def parse_as(tp: Any, text: str) -> Any:
    """Parse the model's final answer as ``tp``. Raises ``OutputInvalid``."""
    return _coerce(extract_json(text), tp, "$")
