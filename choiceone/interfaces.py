"""The AI-side abstraction the decision layer talks to.

``choiceone`` sits between your code and an AI model. Your code only speaks in
typed :class:`~choiceone.primitives.Choice`, :class:`~choiceone.primitives.Score`
and :class:`~choiceone.primitives.Noul` questions; the :class:`Backend` is the
seam where those questions become a concrete model call.

To embed a new AI (a hosted API, a different local model, a mock), subclass
:class:`Backend` and implement three methods:

``build_prompt(user_content)``
    Turn the rendered question text into a prompt string for your model.
``count_tokens(texts)``
    Return the token length of each prompt (used for usage accounting).
``probs_over_labels(prompts, labels_per_prompt)``
    Return, for each prompt, a probability distribution over the given labels.
    This is the only place that touches the model.

The class never generates free text: a decision is a distribution over a closed
set of labels.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Sequence

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"

DEFAULT_SYSTEM_PROMPT = (
    "You are a ChoiceOne model. You make fast, structured decisions about a "
    "state and answer with a single label. Never explain your reasoning."
)

ANSWER_MARKER = "\nAnswer:"

__all__ = [
    "Backend",
    "DEFAULT_MODEL_ID",
    "DEFAULT_SYSTEM_PROMPT",
    "ANSWER_MARKER",
    "render",
    "render_state",
]


def render(value: Any) -> str:
    """Render an instruction/criteria value that may be a str, dict or list."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


class Backend(ABC):
    """Base class for the model side of a ChoiceOne decision layer."""

    model_name: str = "unknown"

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    # -- required ------------------------------------------------------------

    @abstractmethod
    def build_prompt(self, user_content: str) -> str:
        """Render the question body into a prompt string for the model."""

    @abstractmethod
    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Return the token length of each prompt."""

    @abstractmethod
    def probs_over_labels(
        self, prompts: Sequence[str], labels_per_prompt: Sequence[Sequence[str]]
    ) -> list[list[float]]:
        """Return a probability distribution over each prompt's labels."""

    # -- provided ------------------------------------------------------------

    def choice_prompt(
        self,
        state: Any,
        instructions: Any,
        criteria: dict[Any, Any],
        *,
        ordered: bool,
    ) -> str:
        lines = [
            "# State",
            render_state(state),
            "",
            "# Question",
            render(instructions),
            "",
            "# " + ("Levels" if ordered else "Options"),
        ]
        for index, (key, description) in enumerate(criteria.items()):
            text = render(description)
            if isinstance(key, int):
                lines.append(f"{index}: {text}" if text else f"{index}")
            else:
                lines.append(f"{index}: {key}" + (f" - {text}" if text else ""))
        kind = "level" if ordered else "option"
        lines += [
            "",
            "# Answer format",
            f"Reply with only the number of the best {kind}.",
            ANSWER_MARKER,
        ]
        return self.build_prompt("\n".join(lines))

    def noul_prompt(
        self, state: Any, instructions: Any, criteria: dict[str, Any] | None
    ) -> str:
        lines = ["# State", render_state(state), "", "# Question", render(instructions)]
        if criteria:
            lines += ["", "# What yes and no mean"]
            for key in ("true", "false"):
                if key in criteria:
                    lines.append(f"{key}: {render(criteria[key])}")
        lines += [
            "",
            "# Answer format",
            'Reply with only "yes" or "no".',
            ANSWER_MARKER,
        ]
        return self.build_prompt("\n".join(lines))
