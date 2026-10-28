
from uuid import UUID

import logfire
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import answer_question
from app.memory import ConversationMemory

# Send traces to Logfire when a valid token is configured.
logfire.configure(send_to_logfire="if-token-present")

conversation_memory = ConversationMemory()

app = FastAPI(
    title="Enterprise AI Knowledge Platform",
    description="Document-grounded enterprise HR policy assistant",
    version="1.0.0",
)


def safe_request_attributes(request, attributes) -> dict:
    """Avoid recording request arguments such as questions and user IDs."""
    return {}


# Instrument incoming HTTP requests.
logfire.instrument_fastapi(
    app,
    request_attributes_mapper=safe_request_attributes,
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: UUID = Field(
        description="Unique identifier for this conversation"
    )
    user_id: str = Field(..., min_length=1, max_length=255)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        session_id = str(request.session_id)

        # Preserve the existing conversation-history lookup.
        history = conversation_memory.get_history(session_id)

        # Trace the execution time and failures of the RAG operation.
        with logfire.span("rag.answer_question"):
            result = answer_question(
                question=request.question,
                session_id=session_id,
                user_id=request.user_id,
            )

        if isinstance(result, dict):
            answer = result.get("answer")
            if not isinstance(answer, str):
                raise ValueError(
                    "RAG response does not contain a string answer."
                )
        elif isinstance(result, str):
            answer = result
        else:
            raise ValueError("Unexpected RAG response format.")

        conversation_memory.add_turn(
            session_id=session_id,
            question=request.question,
            answer=answer,
        )

        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
