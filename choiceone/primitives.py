"""TypeSafe ChoiceOne question primitives.

Mirrors ``typesafe_sdk`` question types: Choice, Score and Noul.
Each question evaluates one judgment against a shared ``state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Choice:
    """Pick one option from a fixed, unordered set.

    ``criteria`` maps each option key to a description (str, dict, list or
    ``None``). Option names are what get returned in the answer.
    """

    instructions: Any
    criteria: dict[str, Any]
    type: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Choice.criteria must contain at least one option")


@dataclass
class Score:
    """Rate the state along ordered levels (low end first)."""

    instructions: Any
    criteria: list[Any]
    type: str = field(default="score", init=False)

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("Score.criteria needs at least two levels")


@dataclass
class Noul:
    """The probability that a yes/no statement is true, in ``[0, 1]``.

    ``criteria`` is optional; when given it should describe ``true``/``false``.
    """

    instructions: Any
    criteria: dict[str, Any] | None = None
    type: str = field(default="noul", init=False)


Question = Choice | Score | Noul
