"""Wire format for the ``/v1/choiceone`` HTTP endpoint.

Request body (same three top-level fields as the TypeSafe API)::

    {
      "state": "<content to evaluate>",
      "model": "Qwen/Qwen3-0.6B",
      "questions": {
        "department": {
          "type": "choice",
          "instructions": "Which team should handle this?",
          "criteria": {"returns": "...", "shipping": "...", "billing": "..."}
        },
        "is_urgent": {"type": "noul", "instructions": "..."},
        "frustration": {"type": "score", "instructions": "...", "criteria": ["...", "..."]}
      }
    }

Response body matches the TypeSafe API: ``model``, ``answers`` and ``usage``.
"""

from __future__ import annotations

from typing import Any

from .answers import ChoiceOneResponse
from .primitives import Choice, Noul, Question, Score


def parse_questions(raw: dict[str, Any]) -> dict[str, Question]:
    """Build typed question objects from the request's ``questions`` map."""
    questions: dict[str, Question] = {}
    for question_id, spec in raw.items():
        qtype = spec.get("type")
        if "instructions" not in spec:
            raise ValueError(f"question {question_id!r} is missing 'instructions'")
        if qtype == "choice":
            questions[question_id] = Choice(
                instructions=spec["instructions"], criteria=spec["criteria"]
            )
        elif qtype == "score":
            questions[question_id] = Score(
                instructions=spec["instructions"], criteria=spec["criteria"]
            )
        elif qtype == "noul":
            questions[question_id] = Noul(
                instructions=spec["instructions"], criteria=spec.get("criteria")
            )
        else:
            raise ValueError(
                f"question {question_id!r} has unsupported type {qtype!r}; "
                "expected one of 'choice', 'score', 'noul'"
            )
    return questions


def response_to_wire(response: ChoiceOneResponse) -> dict[str, Any]:
    """Serialise a response to the API's JSON shape."""
    return response.model_dump()
