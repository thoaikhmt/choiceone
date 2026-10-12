"""Typed ChoiceOne answers and response envelope."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = field(default="choice", init=False)


@dataclass
class ScoreAnswer:
    score: float
    legend: dict[int, Any]
    probabilities: dict[int, float]
    confidence: float
    type: str = field(default="score", init=False)


@dataclass
class NoulAnswer:
    noul: float
    type: str = field(default="noul", init=False)


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class ChoiceOneResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage

    def model_dump(self) -> dict[str, Any]:
        return _jsonable(self)

    def model_dump_json(self, **kwargs: Any) -> str:
        return json.dumps(self.model_dump(), **kwargs)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
