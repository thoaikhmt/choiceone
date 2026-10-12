"""choiceone: a type-safe decision layer between your code and an AI model.

The TypeSafe-style "ChoiceOne" protocol: unstructured ``state`` in, typed
probabilistic decisions out. Questions (:class:`Choice`, :class:`Score`,
:class:`Noul`) are scored as a distribution over a closed set of labels, so the
model cannot return a type error. Every Choice/Score answer carries a
``confidence`` derived from the distribution's spread.

Embed it anywhere::

    from choiceone import ChoiceOne, Choice

    so = ChoiceOne()  # or ChoiceOne(backend=MyBackend())
    answer = so(state, {\"team\": Choice(\"Which team?\", {\"a\": ..., \"b\": ...})})

    if answer.answers[\"team\"].confidence > 0.8:
        route(answer.answers[\"team\"].choice)

The default backend needs the optional local model dependencies
(``pip install 'choiceone[local]'``). Importing this package does not import
torch; the local backend is loaded only when used.
"""

from .answers import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    ChoiceOneResponse,
    Usage,
)
from .client import ChoiceOne, ChoiceOneClient
from .confidence import choice_confidence, noul_confidence, score_confidence
from .interfaces import DEFAULT_MODEL_ID, Backend
from .primitives import Choice, Noul, Question, Score
from .protocol import parse_questions, response_to_wire
from .remote import OpenAIBackend

__all__ = [
    "Choice",
    "Score",
    "Noul",
    "Question",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "ChoiceOneResponse",
    "Usage",
    "ChoiceOne",
    "ChoiceOneClient",
    "Backend",
    "OpenAIBackend",
    "DEFAULT_MODEL_ID",
    "LocalBackend",
    "choice_confidence",
    "score_confidence",
    "noul_confidence",
    "parse_questions",
    "response_to_wire",
]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("choiceone")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"


def __getattr__(name: str):
    if name == "LocalBackend":
        from .backend import LocalBackend

        return LocalBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
