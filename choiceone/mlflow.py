"""MLflow tracing integration for choiceone.

Usage::

    import mlflow
    from choiceone import ChoiceOne
    import choiceone.mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    choiceone.mlflow.autolog()

    so = ChoiceOne(backend=...)
    so(state, questions)  # every decision is traced

Requires the optional dependency::

    pip install 'choiceone[mlflow]'

The integration records two kinds of span:

* ``choiceone.decision`` (``CHAIN``) wraps every ``ChoiceOne`` call with the
  state, questions, typed answers and token usage.
* ``choiceone.backend`` (``LLM``) wraps the model seam
  (``Backend.probs_over_labels``) with the prompts, the label set and the
  returned distributions.

``mlflow.openai.autolog()`` cannot trace this library: the HTTP backends use
stdlib ``urllib`` rather than the OpenAI SDK. Use this integration instead.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

from .client import ChoiceOneClient
from .interfaces import Backend
from .remote import RemoteBackend

__all__ = ["autolog", "FLAVOR_NAME"]

FLAVOR_NAME = "choiceone"
ROOT_SPAN_NAME = "choiceone.decision"
BACKEND_SPAN_NAME = "choiceone.backend"

_CHAT_USAGE_ATTRIBUTE = getattr(
    SpanAttributeKey, "CHAT_USAGE", "mlflow.chat.tokenUsage"
)
_INPUT_TOKENS = "input_tokens"
_OUTPUT_TOKENS = "output_tokens"
_TOTAL_TOKENS = "total_tokens"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Enable (or disable) MLflow tracing for choiceone.

    Args:
        log_traces: If ``True``, record a trace for every ``ChoiceOne`` call.
            If ``False``, calls pass through untraced. Defaults to ``True``.
        disable: If ``True``, remove the choiceone tracing patches.
            Defaults to ``False``.
        silent: If ``True``, suppress MLflow event logs and warnings emitted
            while the integration is active. Defaults to ``False``.
    """
    safe_patch(FLAVOR_NAME, ChoiceOneClient, "choice_one", _patched_choice_one)
    safe_patch(FLAVOR_NAME, Backend, "probs_over_labels", _patched_probs_over_labels)
    safe_patch(
        FLAVOR_NAME, RemoteBackend, "probs_over_labels", _patched_probs_over_labels
    )
    _patch_local_backend()


def _patch_local_backend() -> None:
    """Patch ``LocalBackend`` without importing torch unless it is already in use."""

    def patch(module: Any) -> None:
        if not _log_traces_enabled():
            return
        backend_class = getattr(module, "LocalBackend", None)
        if backend_class is not None:
            safe_patch(
                FLAVOR_NAME,
                backend_class,
                "probs_over_labels",
                _patched_probs_over_labels,
            )

    if "choiceone.backend" in sys.modules:
        patch(sys.modules["choiceone.backend"])
        return

    try:
        from mlflow.utils.import_hooks import register_post_import_hook
    except Exception:  # pragma: no cover - mlflow without import hooks
        return
    register_post_import_hook(patch, "choiceone.backend", overwrite=False)


def _patched_choice_one(original: Any, self: ChoiceOneClient, state: Any, questions: Any):
    if not _log_traces_enabled():
        return original(self, state, questions)

    with mlflow.start_span(name=ROOT_SPAN_NAME, span_type=SpanType.CHAIN) as span:
        span.set_inputs(
            _serialize(
                {
                    "model": self.model,
                    "state": state,
                    "questions": questions,
                }
            )
        )
        span.set_attribute(SpanAttributeKey.MODEL, self.model)

        response = original(self, state, questions)

        span.set_outputs(_serialize(response))
        _set_usage(span, response)
        return response


def _patched_probs_over_labels(
    original: Any,
    self: Backend,
    prompts: Sequence[str],
    labels_per_prompt: Sequence[Sequence[str]],
):
    if not _log_traces_enabled():
        return original(self, prompts, labels_per_prompt)

    model = getattr(self, "model_name", "unknown")
    with mlflow.start_span(name=BACKEND_SPAN_NAME, span_type=SpanType.LLM) as span:
        span.set_inputs(
            {
                "model": model,
                "prompts": list(prompts),
                "labels": [list(labels) for labels in labels_per_prompt],
            }
        )
        span.set_attribute(SpanAttributeKey.MODEL, model)
        span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, type(self).__name__)

        distributions = original(self, prompts, labels_per_prompt)

        span.set_outputs({"distributions": [list(row) for row in distributions]})
        return distributions


def _log_traces_enabled() -> bool:
    return AutoLoggingConfig.init(flavor_name=FLAVOR_NAME).log_traces


def _set_usage(span: Any, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None and output_tokens is None:
        return

    parsed: dict[str, int] = {}
    if input_tokens is not None:
        parsed[_INPUT_TOKENS] = int(input_tokens)
    if output_tokens is not None:
        parsed[_OUTPUT_TOKENS] = int(output_tokens)
    if input_tokens is not None and output_tokens is not None:
        parsed[_TOTAL_TOKENS] = int(input_tokens) + int(output_tokens)
    span.set_attribute(_CHAT_USAGE_ATTRIBUTE, parsed)


def _serialize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _serialize(dataclasses.asdict(value))

    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _serialize(dump())
        except Exception:
            pass

    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
