# choiceone

A type-safe **decision layer** between your Python code and an AI model.

Instead of generating prose and parsing it back, `choiceone` sends an
unstructured `state` plus a closed set of typed questions and returns typed,
probabilistic decisions your code can branch on directly. Answers cannot have
a type error: they are always a distribution over the options you supplied.

Implements the "ChoiceOne" protocol: **state in, typed probabilistic
decisions out**.

## Install

```bash
pip install choiceone            # core, no model dependencies
pip install 'choiceone[local]'   # + local Hugging Face backend (torch, transformers)
pip install 'choiceone[server]'  # + HTTP server (fastapi, uvicorn)
```

## Quick start

```python
from choiceone import ChoiceOne, Choice

so = ChoiceOne()  # default local model
response = so(
    state="My running shoes arrived in the wrong size.",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={
                "returns": "Exchanges, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
            },
        ),
    },
)

answer = response.answers["department"]
print(answer.choice, answer.probabilities, answer.confidence)
```

## Primitives

| Question | Returns |
|---|---|
| `Choice` | `choice`, `probabilities`, `confidence` |
| `Score`  | `score`, `legend`, `probabilities`, `confidence` |
| `Noul`   | `noul` (probability the statement is true, `0`-`1`) |

Every `Choice`/`Score` answer includes a `confidence` derived from how peaked
the distribution is, so your code can act automatically, ask, or escalate.

## Backends

The model side is a pluggable `Backend`. `choiceone` ships with these:

- `LocalBackend` — a local Hugging Face causal LM (default `Qwen/Qwen3-0.6B`).
  Text-generation-free: reads the next-token distribution.
- `OpenAIBackend` — any OpenAI-compatible `logprobs` endpoint (DeepSeek,
  OpenAI, vLLM, ...). Stdlib only.
- `GoogleBackend` — Google Gemini, reading `logprobs` from `generateContent`.
  Stdlib only.
- `AnthropicBackend` — Claude, which has no logprobs API, so it elicits an
  explicit JSON probability per label via structured output
  (`pip install 'choiceone[anthropic]'`).

The logprobs backends (`OpenAIBackend`, `GoogleBackend`) score a single content
token, so they support up to 10 options and `yes`/`no`.

Embed your own AI by subclassing `Backend` and implementing three methods:

```python
from choiceone import Backend

class MyBackend(Backend):
    def build_prompt(self, user_content): ...
    def count_tokens(self, texts): ...
    def probs_over_labels(self, prompts, labels_per_prompt): ...
```

## MLflow tracing

`choiceone` ships a first-party MLflow integration (`pip install 'choiceone[mlflow]'`).
It records a `choiceone.decision` span for every call, with a nested
`choiceone.backend` LLM span for the model seam:

```python
import mlflow
from choiceone import ChoiceOne
import choiceone.mlflow

mlflow.set_tracking_uri("http://localhost:5000")
choiceone.mlflow.autolog()

so = ChoiceOne(backend=backend)
so(state, questions)  # traced
```

`mlflow.openai.autolog()` does **not** trace `choiceone`: the hosted backends
talk HTTP through stdlib `urllib`, not the OpenAI SDK. Use
`choiceone.mlflow.autolog()` instead. Call `autolog(disable=True)` to remove the
patches, or `autolog(log_traces=False)` to keep the integration configured
without recording traces.

## HTTP server

```bash
choiceone-serve --model Qwen/Qwen3-0.6B --port 8000
# POST /v1/choiceone  {"state": ..., "questions": {...}}
```

## License

MIT
