
from uuid import UUID
import os
import secrets

import logfire
from fastapi import FastAPI, HTTPException, Header, Depends
# from google.oauth2 import id_token
# from google.auth.transport.requests import Request as GoogleRequest
import jwt
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field

from app.rag_pipeline import answer_question
# from app.memory import ConversationMemory

from app.llm.client import extract_long_term_memories
from app.memory_store import (
    store_memory,
    contains_sensitive_data,
)
from contextlib import asynccontextmanager

from app.embeddings import get_embedding_model
from app.retrieval.search import _get_reranker
from app.rag_trace import capture_rag_trace
from app.conversation_store import (
    create_conversation,
    save_message,
    get_conversations,
    get_messages,
    set_conversation_pinned,
    delete_conversation,
)

# Send traces to Logfire when a valid token is configured.
logfire.configure(send_to_logfire="if-token-present")

# GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")



STREAMLIT_AUTH_SECRET = os.getenv("STREAMLIT_AUTH_SECRET")

AUTH_ISSUER = "enterprise-hr-assistant"
AUTH_AUDIENCE = "enterprise-hr-api"


def verify_google_token(
    authorization: str | None = Header(default=None),
) -> str:
    """Verify the signed identity assertion issued by Streamlit."""
    if not authorization:
        raise HTTPException(
        status_code=401,
        detail="Authentication required.",
    )

    if not STREAMLIT_AUTH_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Application authentication is not configured.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header.",
        )

    token = authorization.removeprefix("Bearer ").strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token.",
        )

    try:
        claims = jwt.decode(
            token,
            STREAMLIT_AUTH_SECRET,
            algorithms=["HS256"],
            issuer=AUTH_ISSUER,
            audience=AUTH_AUDIENCE,
            options={
                "require": [
                    "exp",
                    "iat",
                    "sub",
                    "iss",
                    "aud",
                    "email_verified",
                ]
            },
        )

        user_id = claims.get("sub")

        if not user_id:
            raise ValueError("Missing authenticated user identity.")

        if claims.get("email_verified") is not True:
            raise ValueError("User email is not verified.")

        return user_id

    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid user identity.",
        ) from exc

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


# class QuestionRequest(BaseModel):
#     question: str = Field(..., min_length=1, max_length=2000)
#     session_id: UUID = Field(
#         description="Unique identifier for this conversation"
#     )
#     user_id: str = Field(..., min_length=1, max_length=255)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: UUID = Field(
        description="Unique identifier for this conversation"
    )

class CreateConversationRequest(BaseModel):
    conversation_id: UUID | None = None


class PinConversationRequest(BaseModel):
    is_pinned: bool

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/conversations")
def create_new_conversation(
    request: CreateConversationRequest | None = None,
    user_id: str = Depends(verify_google_token),
):
    """Create a new conversation for the authenticated user."""

    try:
        conversation_id = (
            str(request.conversation_id)
            if request and request.conversation_id
            else None
        )

        return create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/conversations")
def list_user_conversations(
    limit: int = 50,
    user_id: str = Depends(verify_google_token),
):
    """Retrieve the authenticated user's conversation history."""

    try:
        return {
            "conversations": get_conversations(
                user_id=user_id,
                limit=limit,
            )
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/conversations/{conversation_id}/messages")
def retrieve_conversation_messages(
    conversation_id: UUID,
    user_id: str = Depends(verify_google_token),
):
    """Retrieve messages belonging to an owned conversation."""

    try:
        return {
            "messages": get_messages(
                conversation_id=str(conversation_id),
                user_id=user_id,
            )
        }

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.patch("/conversations/{conversation_id}/pin")
def update_conversation_pin(
    conversation_id: UUID,
    request: PinConversationRequest,
    user_id: str = Depends(verify_google_token),
):
    """Pin or unpin an owned conversation."""

    updated = set_conversation_pinned(
        conversation_id=str(conversation_id),
        user_id=user_id,
        is_pinned=request.is_pinned,
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Conversation was not found.",
        )

    return {
        "success": True,
        "conversation_id": str(conversation_id),
        "is_pinned": request.is_pinned,
    }


@app.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: UUID,
    user_id: str = Depends(verify_google_token),
):
    """Delete an owned conversation and its messages."""

    deleted = delete_conversation(
        conversation_id=str(conversation_id),
        user_id=user_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation was not found.",
        )

    return {
        "success": True,
        "conversation_id": str(conversation_id),
        "message": "Conversation deleted successfully.",
    }


@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    user_id: str = Depends(verify_google_token),
):
    try:
        session_id = str(request.session_id)

        # Ensure the conversation exists and belongs to this user.
        create_conversation(
            user_id=user_id,
            conversation_id=session_id,
        )

        # Handle explicit long-term memory requests separately.
        if is_memory_eligible(request.question):
            saved_count = save_user_memories(
                user_message=request.question,
                user_id=user_id,
                session_id=session_id,
            )

            if saved_count:
                answer = (
                    f"I've saved {saved_count} "
                    "memory item(s) for future conversations."
                )
            else:
                answer = (
                    "I couldn't identify any safe, useful information "
                    "to remember from that message."
                )

            # Persist the memory request in conversation history.
            save_message(
                conversation_id=session_id,
                user_id=user_id,
                role="user",
                content=request.question,
            )

            save_message(
                conversation_id=session_id,
                user_id=user_id,
                role="assistant",
                content=answer,
                metadata={
                    "memory_saved": bool(saved_count),
                    "memory_count": saved_count,
                    "sources": [],
                    "retrieved_contexts": [],
                },
            )

            return {
                "answer": answer,
                "sources": [],
                "retrieved_contexts": [],
                "memory_saved": bool(saved_count),
                "memory_count": saved_count,
            }

        # Preserve existing conversation-history lookup.
        # history = conversation_memory.get_history(session_id)
        # history = conversation_memory.get_history(user_id, session_id)
        # restore_conversation_memory(user_id, session_id)
        # Capture actual RAG pipeline execution stages.
        with capture_rag_trace() as trace_events:
            with logfire.span("rag.answer_question"):
                result = answer_question(
                    question=request.question,
                    session_id=session_id,
                    user_id=user_id,
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

        # Include trace events in the FastAPI response.
        response_data["trace"] = trace_events

        # Persist the user question.
        save_message(
            conversation_id=session_id,
            user_id=user_id,
            role="user",
            content=request.question,
        )

        # Persist the assistant response and supporting metadata.
        save_message(
            conversation_id=session_id,
            user_id=user_id,
            role="assistant",
            content=answer,
            metadata={
                "sources": response_data.get("sources", []),
                "retrieved_contexts": response_data.get(
                    "retrieved_contexts", []
                ),
                "trace": trace_events,
            },
        )

        return response_data

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
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
