"""Skill crystallization: proven tool sequences become tools themselves.

A trace lake is full of tool-call sequences that demonstrably worked. This
module mines them into **skills** -- macro-tools the agent can call in one
step next time:

    runs = [Run.load(p, agent=agent) for p in glob("runs/*.loom.json")]
    skills = mine(runs)                       # frequent, successful sequences
    toolmap = {t.name: t for t in my_tools}
    agent2 = Agent(model=..., tools=[*my_tools, *[s.as_tool(toolmap) for s in skills]])

Parameterization is learned by comparison: argument values that VARY across
the mined occurrences become the skill's parameters; values that never change
are baked in. A skill executes its underlying real tools in order and is
recorded as a single tool effect -- so replays serve its recorded result
without re-executing anything, like every other tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .providers.base import ModelResponse
from .tools import Tool

_PARAM_MARKER = "$param"


@dataclass
class Skill:
    """A mined (or hand-written) macro-tool: a sequence of proven tool calls.

    ``steps`` hold each underlying call's arguments, where a parameter slot is
    the marker dict ``{"$param": "<name>"}``. ``support`` says how many
    recorded runs this sequence was mined from.
    """

    name: str
    description: str
    steps: list[dict]  # [{"tool": name, "args": {...}}, ...]
    params: list[str] = field(default_factory=list)
    support: int = 0

    def as_tool(self, toolmap: "dict[str, Tool] | list[Tool]") -> Tool:
        """Wrap as a callable Tool, bound to the real tools it orchestrates."""
        tools = (
            {t.name: t for t in toolmap} if isinstance(toolmap, list) else dict(toolmap)
        )
        for step in self.steps:
            if step["tool"] not in tools:
                raise KeyError(f"skill {self.name!r} needs tool {step['tool']!r}")

        def fn(**kwargs: Any) -> str:
            result: Any = ""
            for step in self.steps:
                args = {
                    k: kwargs[v[_PARAM_MARKER]]
                    if isinstance(v, dict) and _PARAM_MARKER in v
                    else v
                    for k, v in step["args"].items()
                }
                result = tools[step["tool"]](**args)
            return str(result)

        return Tool(
            name=self.name,
            description=self.description,
            fn=fn,
            input_schema={
                "type": "object",
                "properties": {p: {} for p in self.params},
                "required": list(self.params),
            },
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "params": self.params,
            "support": self.support,
        }

    @staticmethod
    def from_dict(d: dict) -> "Skill":
        return Skill(**d)


def _call_sequence(run) -> list[tuple[str, dict]]:
    """The ordered top-level tool calls of a run: (tool_name, input_args)."""
    calls: list[tuple[str, dict]] = []
    for e in run.log:
        if e.kind == "model" and e.depth == 0:
            resp = ModelResponse.from_dict(e.result)
            for tc in resp.tool_calls:
                calls.append((tc.name, dict(tc.input)))
    return calls


def mine(runs: list, min_support: int = 2, min_len: int = 2) -> list[Skill]:
    """Extract skills: tool sequences seen in >= ``min_support`` successful runs.

    Only finished runs count (paused/truncated ones prove nothing). Longer
    sequences win over their own sub-sequences.
    """
    sequences = [
        _call_sequence(r) for r in runs if not r.paused and not r.truncated
    ]
    sequences = [s for s in sequences if len(s) >= min_len]

    # Count contiguous n-grams of tool names, remembering each occurrence's args.
    occurrences: dict[tuple, list[list[dict]]] = {}
    for seq in sequences:
        names = [name for name, _ in seq]
        seen_in_this_run: set[tuple] = set()
        for n in range(min_len, len(seq) + 1):
            for start in range(len(seq) - n + 1):
                gram = tuple(names[start : start + n])
                if gram in seen_in_this_run:
                    continue  # support counts runs, not repetitions within one
                seen_in_this_run.add(gram)
                occurrences.setdefault(gram, []).append(
                    [args for _, args in seq[start : start + n]]
                )

    frequent = {g: occ for g, occ in occurrences.items() if len(occ) >= min_support}
    # Prefer maximal sequences: drop grams contained in a longer frequent gram.
    maximal = [
        g
        for g in frequent
        if not any(g != other and _contains(other, g) for other in frequent)
    ]

    skills = []
    for gram in sorted(maximal, key=len, reverse=True):
        skills.append(_parameterize(gram, frequent[gram]))
    return skills


def _contains(longer: tuple, shorter: tuple) -> bool:
    n = len(shorter)
    return any(longer[i : i + n] == tuple(shorter) for i in range(len(longer) - n + 1))


def _parameterize(gram: tuple, occ: list[list[dict]]) -> Skill:
    """Compare argument values across occurrences: varying -> param, constant -> baked."""
    steps: list[dict] = []
    params: list[str] = []
    for i, tool_name in enumerate(gram):
        args: dict = {}
        keys = {k for one in occ for k in one[i]}
        for k in sorted(keys):
            values = [one[i].get(k) for one in occ]
            if all(v == values[0] for v in values):
                args[k] = values[0]  # never varies: baked in
            else:
                pname = k if k not in params else f"{k}_{i}"
                params.append(pname)
                args[k] = {_PARAM_MARKER: pname}
        steps.append({"tool": tool_name, "args": args})

    name = "skill_" + "_then_".join(gram)
    return Skill(
        name=name,
        description=(
            f"Proven sequence mined from {len(occ)} successful runs: "
            + " -> ".join(gram)
            + (f". Parameters: {', '.join(params)}." if params else ".")
        ),
        steps=steps,
        params=params,
        support=len(occ),
    )


def save(skills: list[Skill], path: str) -> None:
    """Persist a skill library as JSON."""
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in skills], f, indent=2)


def load(path: str) -> list[Skill]:
    """Load a skill library saved by ``save``."""
    with open(path) as f:
        return [Skill.from_dict(d) for d in json.load(f)]
