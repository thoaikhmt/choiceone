"""Tests for the local ChoiceOne implementation.

A ``StubBackend`` stands in for the model so the suite runs without loading
weights: it returns predetermined label distributions.
"""

from __future__ import annotations

import json
import math

import pytest

from choiceone import (
    Backend,
    Choice,
    Noul,
    Score,
    ChoiceOne,
    ChoiceOneClient,
    choice_confidence,
    noul_confidence,
    parse_questions,
    response_to_wire,
    score_confidence,
)
from choiceone.server import create_app


class StubBackend(Backend):
    """A minimal custom backend: the only thing an embedder must provide."""

    def __init__(self, distributions):
        super().__init__()
        self.model_name = "stub-model"
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


# -- confidence formulas (values taken from the TypeSafe docs) --------------


def test_score_confidence_matches_docs():
    assert score_confidence([0, 0.57, 0.43]) == pytest.approx(0.355, abs=1e-3)
    assert score_confidence([0, 0.5, 0.5]) == pytest.approx(0.25)
    assert score_confidence([0.5, 0, 0.5]) == pytest.approx(0.0)


def test_choice_confidence_matches_docs():
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-9)


def test_noul_confidence():
    assert noul_confidence(1.0) == 1.0
    assert noul_confidence(0.5) == 0.0
    assert noul_confidence(0.99) == pytest.approx(0.98)


# -- primitives --------------------------------------------------------------


def test_choice_requires_options():
    with pytest.raises(ValueError):
        Choice(instructions="pick", criteria={})


def test_score_requires_two_levels():
    with pytest.raises(ValueError):
        Score(instructions="rate", criteria=["only one"])


def test_type_fields_are_fixed():
    assert Choice(instructions="x", criteria={"a": None}).type == "choice"
    assert Score(instructions="x", criteria=["a", "b"]).type == "score"
    assert Noul(instructions="x").type == "noul"


# -- client assembly ---------------------------------------------------------


def questions():
    return {
        "department": Choice(
            instructions="Which team?",
            criteria={"returns": "d1", "shipping": "d2", "billing": "d3"},
        ),
        "frustration": Score(
            instructions="How frustrated?",
            criteria=["calm", "frustrated", "angry"],
        ),
        "is_urgent": Noul(instructions="Is it urgent?"),
    }


def test_client_builds_typed_answers():
    stub = StubBackend([[0.1, 0.8, 0.1], [0.0, 0.6, 0.4], [0.95, 0.05]])
    client = ChoiceOneClient(backend=stub)

    response = client.choice_one(state="some state", questions=questions())

    department = response.answers["department"]
    assert department.choice == "shipping"
    assert department.probabilities == {"returns": 0.1, "shipping": 0.8, "billing": 0.1}
    assert department.confidence == pytest.approx(choice_confidence([0.1, 0.8, 0.1]))

    frustration = response.answers["frustration"]
    assert frustration.score == pytest.approx(1 * 0.6 + 2 * 0.4)
    assert frustration.probabilities == {0: 0.0, 1: 0.6, 2: 0.4}
    assert frustration.legend[2] == "angry"
    assert frustration.confidence == pytest.approx(score_confidence([0.0, 0.6, 0.4]))

    assert response.answers["is_urgent"].noul == pytest.approx(0.95)
    assert response.model == "stub-model"
    assert response.usage.output_tokens == 3


def test_response_serialises_to_wire_shape():
    stub = StubBackend([[0.1, 0.8, 0.1], [0.0, 0.6, 0.4], [0.95, 0.05]])
    response = ChoiceOneClient(backend=stub).choice_one("s", questions())

    wire = response_to_wire(response)
    assert set(wire) == {"model", "answers", "usage"}
    assert wire["answers"]["department"]["type"] == "choice"
    assert wire["answers"]["frustration"]["legend"] == {
        "0": "calm",
        "1": "frustrated",
        "2": "angry",
    }
    assert wire["answers"]["frustration"]["probabilities"] == {"0": 0.0, "1": 0.6, "2": 0.4}


# -- protocol parsing --------------------------------------------------------


def test_parse_questions_builds_objects():
    parsed = parse_questions(
        {
            "d": {"type": "choice", "instructions": "q", "criteria": {"a": None}},
            "s": {"type": "score", "instructions": "q", "criteria": ["a", "b"]},
            "n": {"type": "noul", "instructions": "q"},
        }
    )
    assert isinstance(parsed["d"], Choice)
    assert isinstance(parsed["s"], Score)
    assert isinstance(parsed["n"], Noul)


def test_parse_questions_rejects_unknown_type():
    with pytest.raises(ValueError):
        parse_questions({"x": {"type": "rank", "instructions": "q"}})


def test_parse_questions_requires_instructions():
    with pytest.raises(ValueError):
        parse_questions({"x": {"type": "noul"}})


# -- HTTP endpoint -----------------------------------------------------------


def test_http_endpoint_round_trip():
    httpx = pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    stub = StubBackend([[0.1, 0.8, 0.1], [0.0, 0.6, 0.4], [0.95, 0.05]])
    app = create_app(client=ChoiceOneClient(backend=stub))

    with TestClient(app) as http:
        payload = {
            "state": "some state",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"returns": "d1", "shipping": "d2", "billing": "d3"},
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated?",
                    "criteria": ["calm", "frustrated", "angry"],
                },
                "is_urgent": {"type": "noul", "instructions": "Is it urgent?"},
            },
        }
        resp = http.post("/v1/choiceone", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["answers"]["department"]["choice"] == "shipping"
    assert body["answers"]["is_urgent"]["noul"] == pytest.approx(0.95)


def test_http_endpoint_rejects_bad_question():
    httpx = pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = create_app(client=ChoiceOneClient(backend=StubBackend([])))
    with TestClient(app) as http:
        resp = http.post(
            "/v1/choiceone",
            json={"state": "s", "questions": {"bad": {"type": "rank"}}},
        )
    assert resp.status_code == 422


# -- multi-token label selection --------------------------------------------


class FakeTokenizer:
    """Maps each character to its own token id; deterministic and dependency free."""

    def __init__(self):
        self.pad_token = 0
        self.eos_token = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": self.encode(text)})()


def test_label_ids_split_on_shared_prefix():
    pytest.importorskip("torch")
    from choiceone.backend import LocalBackend

    backend = LocalBackend.__new__(LocalBackend)
    backend.tokenizer = FakeTokenizer()
    prompt = "abc"
    # label "1" and label "10" diverge after the shared "1" token.
    assert backend._label_ids(prompt, "1", 3) == [ord("1")]
    assert backend._label_ids(prompt, "10", 3) == [ord("1"), ord("0")]


# -- library surface ---------------------------------------------------------


def test_importing_package_does_not_load_torch():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import choiceone
        assert "torch" not in sys.modules, "torch was imported eagerly"
        assert "transformers" not in sys.modules, "transformers was imported eagerly"
        assert "anthropic" not in sys.modules, "anthropic was imported eagerly"
        print("lazy-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lazy-ok" in result.stdout


def test_custom_backend_can_be_embedded():
    stub = StubBackend([[0.9, 0.1]])
    so = ChoiceOneClient(backend=stub)
    response = so(state="s", questions={"n": Noul(instructions="yes?")})
    assert response.answers["n"].noul == pytest.approx(0.9)


def test_choice_one_alias_is_callable():
    stub = StubBackend([[0.2, 0.8]])
    so = ChoiceOne(backend=stub)
    response = so("s", {"d": Choice(instructions="q", criteria={"a": None, "b": None})})
    assert response.answers["d"].choice == "b"


def test_base_backend_builds_choice_prompt():
    stub = StubBackend([])
    prompt = stub.choice_prompt(
        {"k": "v"}, "Which?", {"alpha": "first", "beta": "second"}, ordered=False
    )
    assert "# State" in prompt
    assert "0: alpha - first" in prompt
    assert "1: beta - second" in prompt
    assert prompt.rstrip().endswith("Answer:")


# -- remote (OpenAI-compatible) backend --------------------------------------


def make_remote(canned):
    from choiceone.remote import OpenAIBackend

    backend = OpenAIBackend("m", api_key="test", base_url="http://localhost")
    backend._post = lambda body: canned
    return backend


def test_remote_backend_reads_top_logprobs():
    canned = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "1",
                            "top_logprobs": [
                                {"token": "1", "logprob": -0.043},
                                {"token": "0", "logprob": -3.289},
                            ],
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 42},
    }
    backend = make_remote(canned)
    probs = backend.probs_over_labels(["p"], [["0", "1"]])
    assert probs[0][1] > probs[0][0]
    assert sum(probs[0]) == pytest.approx(1.0)
    assert backend.count_tokens(["p"]) == [42]


def test_remote_backend_case_and_space_insensitive():
    canned = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": " yes",
                            "top_logprobs": [
                                {"token": " yes", "logprob": -0.01},
                                {"token": "No", "logprob": -4.6},
                            ],
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 5},
    }
    backend = make_remote(canned)
    probs = backend.probs_over_labels(["p"], [["yes", "no"]])
    assert probs[0][0] > 0.9


def test_remote_backend_rejects_large_option_sets():
    backend = make_remote({})
    with pytest.raises(NotImplementedError):
        backend.probs_over_labels(["p"], [[str(i) for i in range(12)]])


def test_remote_backend_reports_missing_logprobs():
    canned = {"choices": [{"logprobs": {"content": []}}], "usage": {}}
    backend = make_remote(canned)
    with pytest.raises(RuntimeError, match="no content logprobs"):
        backend._top_logprobs_at_decision("p", ["0", "1"])


# -- remote (Google Gemini) backend ------------------------------------------


def make_google(canned):
    from choiceone.remote import GoogleBackend

    backend = GoogleBackend("gemini-3.1-flash-lite", api_key="test")
    backend._post = lambda body: canned
    return backend


def gemini_reply(mapping, prompt_tokens=11):
    return {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(mapping)}]}}
        ],
        "usageMetadata": {"promptTokenCount": prompt_tokens},
    }


def test_google_backend_reads_structured_probabilities():
    backend = make_google(gemini_reply({"0": 0.25, "1": 0.75}))
    probs = backend.probs_over_labels(["p"], [["0", "1"]])
    assert probs[0][1] > probs[0][0]
    assert sum(probs[0]) == pytest.approx(1.0)
    assert backend.count_tokens(["p"]) == [11]


def test_google_backend_request_uses_response_schema():
    backend = make_google({})
    body = backend._request_body("prompt", ["0", "1"])
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert set(config["responseSchema"]["properties"]) == {"0", "1"}
    # Gemini rejects additionalProperties, so it must be omitted.
    assert "additionalProperties" not in config["responseSchema"]
    assert body["contents"][0]["parts"][0]["text"].startswith("prompt")


def test_google_backend_reports_missing_content():
    backend = make_google({"candidates": [{}], "usageMetadata": {}})
    with pytest.raises(RuntimeError, match="no content"):
        backend.probs_over_labels(["p"], [["0", "1"]])


def test_google_backend_supports_large_option_sets():
    labels = [str(i) for i in range(12)]
    backend = make_google(gemini_reply({label: 1.0 for label in labels}))
    probs = backend.probs_over_labels(["p"], [labels])
    assert len(probs[0]) == 12
    assert sum(probs[0]) == pytest.approx(1.0)


# -- remote (Anthropic) backend ----------------------------------------------


class FakeAnthropicMessages:
    def __init__(self, mapping, input_tokens=9):
        self.mapping = mapping
        self.input_tokens = input_tokens
        self.calls = []

    def create(self, **kwargs):
        import json

        self.calls.append(kwargs)
        block = type("Block", (), {"type": "text", "text": json.dumps(self.mapping)})
        usage = type("Usage", (), {"input_tokens": self.input_tokens})
        return type("Message", (), {"content": [block()], "usage": usage()})


def make_anthropic(mapping):
    from choiceone.remote import AnthropicBackend

    messages = FakeAnthropicMessages(mapping)
    client = type("Client", (), {"messages": messages})()
    backend = AnthropicBackend("claude-opus-5", client=client)
    return backend, messages


def test_anthropic_backend_elicits_and_normalises():
    backend, messages = make_anthropic({"0": 1.0, "1": 3.0})
    probs = backend.probs_over_labels(["p"], [["0", "1"]])
    assert probs[0] == [pytest.approx(0.25), pytest.approx(0.75)]
    assert backend.count_tokens(["p"]) == [9]
    assert messages.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_backend_schema_lists_labels():
    backend, messages = make_anthropic({"yes": 0.9, "no": 0.1})
    backend.probs_over_labels(["p"], [["yes", "no"]])
    schema = messages.calls[0]["output_config"]["format"]["schema"]
    assert set(schema["properties"]) == {"yes", "no"}
    assert schema["required"] == ["yes", "no"]


def test_anthropic_backend_requires_probability_mass():
    backend, _ = make_anthropic({"0": 0.0, "1": 0.0})
    with pytest.raises(RuntimeError, match="no probability mass"):
        backend.probs_over_labels(["p"], [["0", "1"]])
