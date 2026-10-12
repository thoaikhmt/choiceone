"""Remote backends: hosted models score a closed label set into a distribution.

None of these generate free text that has to be parsed for an answer. Two
mechanisms are offered:

* :class:`OpenAIBackend` reads the model's token logprobs at the decision
  position and renormalises them over the question's closed label set. Exact
  and cheap (one request per question), stdlib only (``urllib``).
* :class:`GoogleBackend` and :class:`AnthropicBackend` have no usable logprobs
  API, so they ask for an explicit JSON probability per label via structured
  output and renormalise the result. ``GoogleBackend`` is stdlib only;
  ``AnthropicBackend`` uses the optional ``anthropic`` SDK
  (``pip install 'choiceone[anthropic]'``).

Single-token labels only (logprobs backend)
-------------------------------------------
:class:`OpenAIBackend` scores the *first* content token, so it supports up to 10
options (indices ``0``-``9``) and ``yes``/``no``. Large option sets are handled
by the elicited backends and by the local backend's sequence scoring; asking the
logprobs backend for them raises :class:`NotImplementedError`.

Example::

    from choiceone import ChoiceOne
    from choiceone.remote import OpenAIBackend

    backend = OpenAIBackend(
        "deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        extra_body={"thinking": {"type": "disabled"}},  # disable reasoning
    )
    so = ChoiceOne(backend=backend)
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .interfaces import DEFAULT_SYSTEM_PROMPT, Backend

__all__ = ["OpenAIBackend", "GoogleBackend", "AnthropicBackend"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_PROBABILITY_INSTRUCTION = (
    "\n\nIgnore the answer format above. Instead give the probability that each "
    "label is the correct answer, as a JSON object mapping each label to a number "
    "between 0 and 1. The numbers should sum to 1."
)


class RemoteBackend(Backend):
    """Shared plumbing for backends that make one request per question."""

    def __init__(
        self,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 60.0,
        max_workers: int = 8,
    ) -> None:
        super().__init__(system_prompt=system_prompt)
        self.timeout = timeout
        self.max_workers = max_workers
        self._token_cache: dict[str, int] = {}

    # -- Backend interface ---------------------------------------------------

    def build_prompt(self, user_content: str) -> str:
        return user_content

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [self._token_cache.get(text, max(1, len(text) // 4)) for text in texts]

    def probs_over_labels(
        self, prompts: Sequence[str], labels_per_prompt: Sequence[Sequence[str]]
    ) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._distribution_for, prompts, labels_per_prompt))

    @abstractmethod
    def _distribution_for(self, prompt: str, labels: Sequence[str]) -> list[float]:
        """Return a normalised distribution over ``labels`` for one prompt."""


class _LogprobsBackend(RemoteBackend):
    """Backend that scores the first content token's logprobs."""

    def _distribution_for(self, prompt: str, labels: Sequence[str]) -> list[float]:
        self._check_single_token(labels)
        return self._distribution(self._top_logprobs_at_decision(prompt, labels), labels)

    @abstractmethod
    def _top_logprobs_at_decision(
        self, prompt: str, labels: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Return ``{"token", "logprob"}`` entries at the decision position."""

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _check_single_token(labels: Sequence[str]) -> None:
        for label in labels:
            if label.isdigit() and len(label) > 1:
                raise NotImplementedError(
                    "the logprobs backend scores one token, so it supports at "
                    "most 10 options; use the Google, Anthropic or local "
                    "backend for larger option sets"
                )

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    def _distribution(
        self, top_logprobs: Sequence[dict[str, Any]], labels: Sequence[str]
    ) -> list[float]:
        mass = {self._normalize(label): 0.0 for label in labels}
        for token in top_logprobs:
            key = self._normalize(token["token"])
            if key in mass:
                mass[key] = max(mass[key], math.exp(float(token["logprob"])))
        values = [mass[self._normalize(label)] for label in labels]
        total = sum(values)
        if total <= 0:
            raise RuntimeError(
                f"none of the labels {list(labels)} appeared in the model's "
                "top logprobs; the answer could not be scored"
            )
        return [value / total for value in values]


class OpenAIBackend(_LogprobsBackend):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        top_logprobs: int = 20,
        temperature: float = 1.0,
        max_tokens: int = 4,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt, timeout=timeout, max_workers=max_workers
        )
        self.model_name = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key: pass api_key= or set OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.top_logprobs = top_logprobs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    # -- internals -----------------------------------------------------------

    def _request_body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        body.update(self.extra_body)
        return body

    def _top_logprobs_at_decision(
        self, prompt: str, labels: Sequence[str]
    ) -> list[dict[str, Any]]:
        data = self._post(self._request_body(prompt))
        usage = data.get("usage") or {}
        if usage.get("prompt_tokens"):
            self._token_cache[prompt] = int(usage["prompt_tokens"])

        choice = data["choices"][0]
        content_logprobs = (choice.get("logprobs") or {}).get("content") or []
        if not content_logprobs:
            raise RuntimeError(
                "model returned no content logprobs (is reasoning enabled? "
                "for reasoning models disable thinking, e.g. "
                'extra_body={"thinking": {"type": "disabled"}})'
            )
        return content_logprobs[0].get("top_logprobs") or []

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(
                f"{self.model_name} request failed: HTTP {error.code} {detail}"
            ) from error


class _ElicitedBackend(RemoteBackend):
    """Backend that asks the model for an explicit probability per label."""

    def build_prompt(self, user_content: str) -> str:
        return user_content + _PROBABILITY_INSTRUCTION

    def _distribution_for(self, prompt: str, labels: Sequence[str]) -> list[float]:
        probabilities = self._probabilities(prompt, labels)
        values = [
            max(0.0, float(probabilities.get(label, 0.0))) for label in labels
        ]
        total = sum(values)
        if total <= 0:
            raise RuntimeError(
                f"the model returned no probability mass for the labels "
                f"{list(labels)}"
            )
        return [value / total for value in values]

    @abstractmethod
    def _probabilities(self, prompt: str, labels: Sequence[str]) -> dict[str, float]:
        """Return a mapping of label to an (un-normalised) probability."""

    @staticmethod
    def _schema(labels: Sequence[str], *, strict: bool = True) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {label: {"type": "number"} for label in labels},
            "required": list(labels),
        }
        if strict:
            schema["additionalProperties"] = False
        return schema


class GoogleBackend(_ElicitedBackend):
    """Google Gemini, eliciting a label distribution via structured output.

    Stdlib only (``urllib``). Reads the JSON probabilities from the
    ``responseSchema``-constrained candidate and renormalises them.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_GOOGLE_BASE_URL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt, timeout=timeout, max_workers=max_workers
        )
        self.model_name = model.removeprefix("models/")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not self.api_key:
            raise ValueError(
                "No API key: pass api_key= or set GOOGLE_API_KEY / GEMINI_API_KEY"
            )
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    # -- internals -----------------------------------------------------------

    def _request_body(
        self, prompt: str, labels: Sequence[str]
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
            "responseMimeType": "application/json",
            # Gemini's responseSchema rejects additionalProperties, so relabel
            # the schema for this provider.
            "responseSchema": self._schema(labels, strict=False),
        }
        generation_config.update(self.extra_body)
        return {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

    def _probabilities(self, prompt: str, labels: Sequence[str]) -> dict[str, float]:
        data = self._post(self._request_body(prompt, labels))
        usage = data.get("usageMetadata") or {}
        if usage.get("promptTokenCount"):
            self._token_cache[prompt] = int(usage["promptTokenCount"])

        parts = data["candidates"][0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text:
            raise RuntimeError("model returned no content for the probabilities")
        return {str(key): float(value) for key, value in json.loads(text).items()}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/models/{self.model_name}:generateContent",
            data=json.dumps(body).encode(),
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(
                f"{self.model_name} request failed: HTTP {error.code} {detail}"
            ) from error


class AnthropicBackend(_ElicitedBackend):
    """Claude, eliciting a label distribution via structured JSON output.

    Requires the optional dependency::

        pip install 'choiceone[anthropic]'
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            system_prompt=system_prompt, timeout=timeout, max_workers=max_workers
        )
        self.model_name = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover - optional dependency
                raise ImportError(
                    "AnthropicBackend requires the optional dependency: "
                    "pip install 'choiceone[anthropic]'"
                ) from error
            self._client = anthropic.Anthropic(
                api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    # -- internals -----------------------------------------------------------

    def _probabilities(self, prompt: str, labels: Sequence[str]) -> dict[str, float]:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": self._schema(labels)}
            },
        )
        usage = getattr(message, "usage", None)
        if getattr(usage, "input_tokens", None):
            self._token_cache[prompt] = int(usage.input_tokens)

        text = next(block.text for block in message.content if block.type == "text")
        return {str(key): float(value) for key, value in json.loads(text).items()}
