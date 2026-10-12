"""Local Hugging Face backend.

Scores questions on a local causal LM (default Qwen3-0.6B) by reading the
next-token distribution at a single decision position, so no prose is ever
generated. Requires the optional ``local`` extra::

    pip install 'choiceone[local]'

Two scoring paths are used:

* **single-token** labels (digits ``0``-``9``, ``yes``/``no``): read the
  decision-position logits directly. Exact and cheap (one forward pass).
* **multi-token** labels (option indices ``10`` and up): score each label as a
  token sequence, reusing shared prefixes. An approximation, but it keeps
  answers a proper distribution over large option sets.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interfaces import DEFAULT_MODEL_ID, DEFAULT_SYSTEM_PROMPT, Backend

__all__ = ["LocalBackend", "DEFAULT_MODEL_ID"]


class LocalBackend(Backend):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ) -> None:
        super().__init__(system_prompt=system_prompt)
        self.model_name = model_id
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device
        )
        self.model.eval()

    # -- Backend interface ---------------------------------------------------

    def build_prompt(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [
            len(self.tokenizer(text, add_special_tokens=False).input_ids)
            for text in texts
        ]

    def probs_over_labels(
        self, prompts: Sequence[str], labels_per_prompt: Sequence[Sequence[str]]
    ) -> list[list[float]]:
        prompt_lens = self.count_tokens(prompts)
        label_lens = [
            [len(self._label_ids(text, lab, plen)) for lab in labels]
            for text, plen, labels in zip(prompts, prompt_lens, labels_per_prompt)
        ]
        all_single = all(length == 1 for lengths in label_lens for length in lengths)
        if all_single:
            return self._probs_single_token(prompts, labels_per_prompt, prompt_lens)
        return self._probs_sequence(prompts, labels_per_prompt, prompt_lens)

    # -- internals -----------------------------------------------------------

    def _label_ids(self, text: str, label: str, prompt_len: int) -> list[int]:
        ids = self.tokenizer(text + label, add_special_tokens=False).input_ids
        delta = ids[prompt_len:]
        if delta:
            return delta
        return self.tokenizer(label, add_special_tokens=False).input_ids

    def _probs_single_token(
        self,
        prompts: Sequence[str],
        labels_per_prompt: Sequence[Sequence[str]],
        prompt_lens: Sequence[int],
    ) -> list[list[float]]:
        enc = self.tokenizer(list(prompts), padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1).cpu()

        results: list[list[float]] = []
        for row, (text, labels, plen) in enumerate(
            zip(prompts, labels_per_prompt, prompt_lens)
        ):
            ids = [self._label_ids(text, lab, plen)[0] for lab in labels]
            selected = logprobs[row][torch.tensor(ids)]
            results.append(torch.softmax(selected, dim=-1).tolist())
        return results

    def _next_token_logprobs(self, sequences: list[list[int]]) -> torch.Tensor:
        """Log-softmax of the next token for each (left-padded) token sequence."""
        enc = self.tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, -1, :]
        return torch.log_softmax(logits.float(), dim=-1).cpu()

    def _probs_sequence(
        self,
        prompts: Sequence[str],
        labels_per_prompt: Sequence[Sequence[str]],
        prompt_lens: Sequence[int],
    ) -> list[list[float]]:
        """Score labels as token sequences, reusing shared prefixes.

        The prompt is scored once per question; only the branching parts of the
        label strings (for example the ``1`` shared by ``10``-``14``) get extra
        batched passes. This keeps large option sets far cheaper than running a
        full forward per option.
        """
        prompt_ids = [
            self.tokenizer(text, add_special_tokens=False).input_ids for text in prompts
        ]
        label_ids = [
            [self._label_ids(text, label, plen) for label in labels]
            for text, plen, labels in zip(prompts, prompt_lens, labels_per_prompt)
        ]

        first = self._next_token_logprobs(prompt_ids)
        scores = [
            [float(first[i, ids[0]]) for ids in label_ids[i]]
            for i in range(len(prompts))
        ]

        max_depth = max((len(ids) for group in label_ids for ids in group), default=1)
        for depth in range(2, max_depth + 1):
            nodes: dict[tuple[int, tuple[int, ...]], list[int]] = {}
            for i, group in enumerate(label_ids):
                for j, ids in enumerate(group):
                    if len(ids) >= depth:
                        nodes.setdefault((i, tuple(ids[: depth - 1])), []).append(j)
            if not nodes:
                break

            keys = list(nodes)
            sequences = [prompt_ids[i] + list(prefix) for i, prefix in keys]
            next_logprobs = self._next_token_logprobs(sequences)
            for n, (i, prefix) in enumerate(keys):
                for j in nodes[(i, prefix)]:
                    scores[i][j] += float(next_logprobs[n, label_ids[i][j][depth - 1]])

        return [
            torch.softmax(torch.tensor(group), dim=-1).tolist() for group in scores
        ]
