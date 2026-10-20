"""Tests for the optional MLflow tracing integration."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

import choiceone.mlflow as choiceone_mlflow  # noqa: E402
from choiceone import Backend, Choice, ChoiceOne, Noul, Score  # noqa: E402
from choiceone.remote import RemoteBackend  # noqa: E402


class FixedBackend(RemoteBackend):
    """Remote-style backend whose model call is deterministic and offline."""

    def __init__(self, distributions):
        super().__init__(max_workers=2)
        self.model_name = "fixed-model"
        self._distributions = list(distributions)
        self._index = 0

    def _distribution_for(self, prompt, labels):
        distribution = self._distributions[self._index]
        self._index += 1
        return list(distribution)


class CustomBackend(Backend):
    """Custom backend that overrides the model seam directly."""

    def __init__(self, distributions):
        super().__init__()
        self.model_name = "custom-model"
        self._distributions = list(distributions)
        self._index = 0

    def build_prompt(self, user_content):
        return user_content

    def count_tokens(self, texts):
        return [len(text.split()) for text in texts]

    def probs_over_labels(self, prompts, labels_per_prompt):
        result = self._distributions[self._index : self._index + len(prompts)]
        self._index += len(prompts)
        return result


@pytest.fixture
def mlflow_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    experiment = f"choiceone-test-{uuid.uuid4().hex}"
    mlflow.set_experiment(experiment)
    experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id
    yield experiment_id
    choiceone_mlflow.autolog(disable=True)


def _questions():
    return {
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"returns": "Exchanges", "billing": "Payment problems"},
        ),
        "urgency": Score(
            instructions="How urgent is this?",
            criteria=["Can wait", "Same day"],
        ),
        "needs_refund": Noul(instructions="Is a refund requested?"),
    }


def test_autolog_traces_decision_and_backend(mlflow_env):
    choiceone_mlflow.autolog()

    backend = FixedBackend([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
    so = ChoiceOne(backend=backend)

    response = so("My shoes arrived in the wrong size.", _questions())

    mlflow.flush_trace_async_logging()
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    trace = mlflow.get_trace(trace_id)
    spans = {span.name: span for span in trace.data.spans}

    assert set(spans) == {
        choiceone_mlflow.ROOT_SPAN_NAME,
        choiceone_mlflow.BACKEND_SPAN_NAME,
    }

    root = spans[choiceone_mlflow.ROOT_SPAN_NAME]
    assert root.span_type == "CHAIN"
    assert root.inputs["state"] == "My shoes arrived in the wrong size."
    assert set(root.inputs["questions"]) == set(_questions())
    assert root.outputs["answers"]["department"]["choice"] == "returns"
    assert root.outputs["usage"] == {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    assert root.attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
    }

    child = spans[choiceone_mlflow.BACKEND_SPAN_NAME]
    assert child.span_type == "LLM"
    assert child.parent_id == root.span_id
    assert child.inputs["model"] == "fixed-model"
    assert child.inputs["labels"] == [["0", "1"], ["0", "1"], ["yes", "no"]]
    assert len(child.outputs["distributions"]) == 3


def test_autolog_traces_custom_backends_at_decision_level(mlflow_env):
    choiceone_mlflow.autolog()

    backend = CustomBackend([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    so = ChoiceOne(backend=backend)
    so.decide("My shoes arrived in the wrong size.", _questions())

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    names = {span.name for span in trace.data.spans}

    assert choiceone_mlflow.ROOT_SPAN_NAME in names


def test_autolog_can_be_disabled(mlflow_env):
    choiceone_mlflow.autolog()
    choiceone_mlflow.autolog(disable=True)

    backend = FixedBackend([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    ChoiceOne(backend=backend)("state", _questions())

    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(experiment_ids=[mlflow_env], return_type="list")
    assert traces == []
