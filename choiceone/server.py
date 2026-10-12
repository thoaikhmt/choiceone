"""FastAPI server exposing the ChoiceOne protocol at ``POST /v1/choiceone``.

Run it with::

    python -m choiceone.server --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000

The request/response JSON matches the TypeSafe API. Interactive docs are served
at ``/docs``.
"""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .client import ChoiceOneClient
from .protocol import parse_questions, response_to_wire


class ChoiceOneRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str | None = None


def create_app(client: ChoiceOneClient | None = None, model: str | None = None) -> FastAPI:
    app = FastAPI(title="ChoiceOne", version="0.1.0")
    app.state.client = client
    app.state.model = model

    def get_client() -> ChoiceOneClient:
        if app.state.client is None:
            app.state.client = ChoiceOneClient(app.state.model or "Qwen/Qwen3-0.6B")
        return app.state.client

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/choiceone")
    def choice_one(request: ChoiceOneRequest) -> dict[str, Any]:
        try:
            questions = parse_questions(request.questions)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        response = get_client().choice_one(state=request.state, questions=questions)
        return response_to_wire(response)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ChoiceOne protocol.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(model=args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
