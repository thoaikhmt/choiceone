"""The decision layer: state + typed questions in, typed answers out.

This is the object you embed between your code and an AI model::

    from choiceone import ChoiceOne, Choice

    so = ChoiceOne()  # default local backend; or pass backend=...
    response = so(state="...", questions={"team": Choice(...)})
    response.answers["team"].choice

It is backend agnostic. Pass any :class:`~choiceone.interfaces.Backend` to
``backend=`` to target a hosted API, another local model, or a test double.
"""

from __future__ import annotations

from typing import Any

from .answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    ChoiceOneResponse,
    Usage,
)
from .confidence import choice_confidence, score_confidence
from .interfaces import DEFAULT_MODEL_ID, Backend
from .primitives import Choice, Noul, Question, Score

__all__ = ["ChoiceOneClient", "ChoiceOne"]


class ChoiceOneClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL_ID,
        *,
        backend: Backend | None = None,
        **backend_kwargs: Any,
    ) -> None:
        if backend is None:
            from .backend import LocalBackend

            backend = LocalBackend(model, **backend_kwargs)
        self.backend = backend

    def __enter__(self) -> "ChoiceOneClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        del self.backend

    @property
    def model(self) -> str:
        return getattr(self.backend, "model_name", "unknown")

    def choice_one(
        self, state: Any, questions: dict[str, Question]
    ) -> ChoiceOneResponse:
        """Evaluate every question against ``state`` and return typed answers."""
        prompts: list[str] = []
        labels_per_prompt: list[list[str]] = []

        for question in questions.values():
            if isinstance(question, Choice):
                labels = [str(i) for i in range(len(question.criteria))]
                prompts.append(
                    self.backend.choice_prompt(
                        state, question.instructions, question.criteria, ordered=False
                    )
                )
            elif isinstance(question, Score):
                labels = [str(i) for i in range(len(question.criteria))]
                prompts.append(
                    self.backend.choice_prompt(
                        state,
                        question.instructions,
                        dict(enumerate(question.criteria)),
                        ordered=True,
                    )
                )
            elif isinstance(question, Noul):
                labels = ["yes", "no"]
                prompts.append(
                    self.backend.noul_prompt(
                        state, question.instructions, question.criteria
                    )
                )
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported question type: {type(question)!r}")
            labels_per_prompt.append(labels)

        if not prompts:
            return ChoiceOneResponse(model=self.model, answers={}, usage=Usage(0, 0))

        distributions = self.backend.probs_over_labels(prompts, labels_per_prompt)

        answers: dict[str, Answer] = {}
        for (question_id, question), probability in zip(
            questions.items(), distributions
        ):
            if isinstance(question, Choice):
                keys = list(question.criteria)
                probs = {key: probability[i] for i, key in enumerate(keys)}
                winner = max(probs, key=probs.get)
                answers[question_id] = ChoiceAnswer(
                    choice=winner,
                    probabilities=probs,
                    confidence=choice_confidence(list(probability)),
                )
            elif isinstance(question, Score):
                probs = {i: probability[i] for i in range(len(probability))}
                legend = {i: level for i, level in enumerate(question.criteria)}
                score = sum(i * p for i, p in probs.items())
                answers[question_id] = ScoreAnswer(
                    score=score,
                    legend=legend,
                    probabilities=probs,
                    confidence=score_confidence(list(probability)),
                )
            else:
                answers[question_id] = NoulAnswer(noul=probability[0])

        input_tokens = sum(self.backend.count_tokens(prompts))
        usage = Usage(input_tokens=input_tokens, output_tokens=len(prompts))
        return ChoiceOneResponse(model=self.model, answers=answers, usage=usage)

    # -- embedding conveniences ---------------------------------------------

    def decide(
        self, state: Any, questions: dict[str, Question]
    ) -> ChoiceOneResponse:
        return self.choice_one(state, questions)

    def __call__(
        self, state: Any, questions: dict[str, Question]
    ) -> ChoiceOneResponse:
        return self.choice_one(state, questions)


ChoiceOne = ChoiceOneClient
