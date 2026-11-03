
from uuid import UUID
import os
import secrets

import logfire
from fastapi import FastAPI, HTTPException,Header
from pydantic import BaseModel, Field

from app.rag_pipeline import answer_question
from app.memory import ConversationMemory

from app.llm.client import extract_long_term_memories
from app.memory_store import (
    store_memory,
    contains_sensitive_data,
)
from contextlib import asynccontextmanager

from app.embeddings import get_embedding_model
from app.retrieval.search import _get_reranker
from app.rag_trace import capture_rag_trace

# Send traces to Logfire when a valid token is configured.
logfire.configure(send_to_logfire="if-token-present")

conversation_memory = ConversationMemory()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload ML models when FastAPI starts."""

    with logfire.span("application.model_preloading"):

        logfire.info("Loading Sentence Transformer embedding model")
        get_embedding_model()

        logfire.info("Loading CrossEncoder reranker")
        _get_reranker()

    logfire.info("All ML models loaded successfully")

    yield

    logfire.info("Enterprise AI Knowledge Platform shutting down")

app = FastAPI(
    title="Enterprise AI Knowledge Platform",
    description="Document-grounded enterprise HR policy assistant",
    version="1.0.0",
    lifespan=lifespan,
)


def safe_request_attributes(request, attributes) -> dict:
    """Avoid recording request arguments such as questions and user IDs."""
    return {}


def is_memory_eligible(user_message: str) -> bool:
    """Allow explicit memory requests or durable personal/project statements."""

    text = user_message.strip().lower()

    explicit_memory_phrases = (
        "remember that",
        "remember this",
        "save this",
        "keep in mind",
        "don't forget",
    )

    return any(
        phrase in text
        for phrase in explicit_memory_phrases
    )

def save_user_memories(
    user_message: str,
    user_id: str,
    session_id: str,
) -> int:
    """Extract and save safe, explicitly user-provided memories."""
    if not is_memory_eligible(user_message):
        return 0

    extracted_memories = extract_long_term_memories(
        question=user_message,
        answer="",
    )

    saved_count = 0

    for item in extracted_memories:
        memory_text = item["memory"]
        memory_type = item["type"]

        if contains_sensitive_data(memory_text):
            continue

        store_memory(
            user_id=user_id,
            session_id=session_id,
            memory_text=memory_text,
            memory_type=memory_type,
        )

        saved_count += 1

    return saved_count


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

        # Handle explicit long-term memory requests separately.
        if is_memory_eligible(request.question):
            saved_count = save_user_memories(
                user_message=request.question,
                user_id=request.user_id,
                session_id=session_id,
            )

            if saved_count:
                return {
                    "answer": (
                        f"I've saved {saved_count} "
                        "memory item(s) for future conversations."
                    ),
                    "sources": [],
                    "retrieved_contexts": [],
                    "memory_saved": True,
                    "memory_count": saved_count,
                }

            return {
                "answer": (
                    "I couldn't identify any safe, useful information "
                    "to remember from that message."
                ),
                "sources": [],
                "retrieved_contexts": [],
                "memory_saved": False,
                "memory_count": 0,
            }

        # Preserve existing conversation-history lookup.
        history = conversation_memory.get_history(session_id)

        # Capture actual RAG pipeline execution stages.
        with capture_rag_trace() as trace_events:
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

            response_data = dict(result)

        elif isinstance(result, str):
            answer = result

            response_data = {
                "answer": answer,
                "sources": [],
                "retrieved_contexts": [],
            }

        else:
            raise ValueError("Unexpected RAG response format.")

        conversation_memory.add_turn(
            session_id=session_id,
            question=request.question,
            answer=answer,
        )

        # Include trace events in the FastAPI response.
        response_data["trace"] = trace_events

        return response_data

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

class EvaluationRequest(BaseModel):
    question_count: int = Field(default=1, ge=1, le=5)

import time

@app.post("/evaluate")
def evaluate_ragas(
    request: EvaluationRequest,
    x_evaluation_token: str = Header(...),
):
    expected_token = os.getenv("EVALUATION_API_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Evaluation authentication is not configured.",
        )

    if not secrets.compare_digest(
        x_evaluation_token,
        expected_token,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid evaluation token.",
        )

    start_time = time.perf_counter()

    try:
        logfire.info(
            "RAGAS evaluation started",
            question_count=request.question_count,
        )

        from evals.run_ragas import run_evaluation

        import_start = time.perf_counter()

        logfire.info(
            "RAGAS evaluation module loaded",
            elapsed_seconds=round(
                time.perf_counter() - import_start, 2
            ),
        )

        with logfire.span("rag.evaluation"):
            result = run_evaluation(request.question_count)

        elapsed = round(time.perf_counter() - start_time, 2)

        logfire.info(
            "RAGAS evaluation completed",
            elapsed_seconds=elapsed,
        )

        return {
            "status": "completed",
            "elapsed_seconds": elapsed,
            "results": result,
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
