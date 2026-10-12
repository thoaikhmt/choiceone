"""OpenAI-compatible remote backend.

Talks to any OpenAI-style ``/chat/completions`` endpoint that returns
``logprobs`` (DeepSeek, OpenAI, vLLM, LM Studio, ...). It reads the
distribution at the first content token and renormalises it over the question's
closed label set, so the answer is a proper probability distribution.

No third-party dependency: it uses ``urllib`` from the standard library and
issues the per-question requests concurrently.

Single-token labels only
------------------------
The remote path scores the *first* content token, so it supports up to 10
options (indices ``0``-``9``) and ``yes``/``no``. Large option sets are handled
by the local backend's sequence scoring; asking the remote backend for them
raises :class:`NotImplementedError`.

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .interfaces import DEFAULT_SYSTEM_PROMPT, Backend

__all__ = ["OpenAIBackend"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIBackend(Backend):
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
        super().__init__(system_prompt=system_prompt)
        self.model_name = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key: pass api_key= or set OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.top_logprobs = top_logprobs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self.max_workers = max_workers
        self._token_cache: dict[str, int] = {}

    # -- Backend interface ---------------------------------------------------

    def build_prompt(self, user_content: str) -> str:
        return user_content

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [self._token_cache.get(t, max(1, len(t) // 4)) for t in texts]

    def probs_over_labels(
        self, prompts: Sequence[str], labels_per_prompt: Sequence[Sequence[str]]
    ) -> list[list[float]]:
        for prompts_labels in labels_per_prompt:
            self._check_single_token(prompts_labels)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            top = list(pool.map(self._top_logprobs_at_decision, prompts, labels_per_prompt))
        return [
            self._distribution(top_logprobs, labels)
            for top_logprobs, labels in zip(top, labels_per_prompt)
        ]

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _check_single_token(labels: Sequence[str]) -> None:
        for label in labels:
            if label.isdigit() and len(label) > 1:
                raise NotImplementedError(
                    "remote backend scores one token, so it supports at most "
                    "10 options; use the local backend for larger option sets"
                )

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
