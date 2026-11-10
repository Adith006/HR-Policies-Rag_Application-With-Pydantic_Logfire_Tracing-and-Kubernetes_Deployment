
from app.db.models import ConversationMemory
from app.db.session import SessionLocal
from app.embeddings import get_embedding_model
from sqlalchemy import delete, select
import re

DUPLICATE_THRESHOLD = 0.90
RELEVANCE_THRESHOLD = 0.45

ALLOWED_MEMORY_TYPES = {
    "hr_conversation_summary",
    "user_preference",
    "project_context",
    "user_provided_fact",
    "conversation_context",
}

DEFAULT_MEMORY_TYPES = (
    "hr_conversation_summary",
)

SENSITIVE_PATTERNS = [
    # Database connection strings
    re.compile(
        r"(postgresql|postgres|mysql|mongodb|redis)://\S+",
        re.IGNORECASE,
    ),

    # API keys and access tokens
    re.compile(
        r"\b(api[_-]?key|access[_-]?token|secret[_-]?key)"
        r"\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    # Passwords and credentials
    re.compile(
        r"\b(password|passwd|pwd|client_secret)"
        r"\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    # Private keys
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        re.IGNORECASE,
    ),

    # Common GitHub personal access tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
]


def contains_sensitive_data(text: str) -> bool:
    """Detect common secrets and credentials in memory text."""

    return any(
        pattern.search(text)
        for pattern in SENSITIVE_PATTERNS
    )

def store_memory(
    user_id: str,
    session_id: str,
    memory_text: str,
    memory_type: str,
    metadata: dict | None = None,
) -> int:
    """Store categorized user-specific memories."""

    if not user_id.strip() or not session_id.strip():
        raise ValueError("user_id and session_id are required")

    if memory_type not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"Invalid memory type: {memory_type}")

    memory_text = memory_text.strip()

    if not memory_text:
        raise ValueError("Memory text cannot be empty")

    if contains_sensitive_data(memory_text):
        raise ValueError(
            "Memory rejected: sensitive information detected."
            )

    embedding = get_embedding_model().encode(
        memory_text,
        normalize_embeddings=True,
    ).tolist()

    distance = ConversationMemory.embedding.cosine_distance(
        embedding
    )

    memory_category = ConversationMemory.metadata_["type"].astext

    statement = (
        select(ConversationMemory, distance.label("distance"))
        .where(
            ConversationMemory.user_id == user_id,
            memory_category == memory_type,
        )
        .order_by(distance)
        .limit(1)
    )

    with SessionLocal() as session:
        existing = session.execute(statement).first()

        if existing:
            existing_memory, distance_value = existing
            similarity = 1 - distance_value

            if similarity >= DUPLICATE_THRESHOLD:
                print(
                    "Duplicate memory skipped:",
                    existing_memory.id,
                    f"(similarity={similarity:.4f})",
                )
                return existing_memory.id

        memory = ConversationMemory(
            user_id=user_id,
            session_id=session_id,
            memory_text=memory_text,
            embedding=embedding,
            metadata_={
                **(metadata or {}),
                "type": memory_type,
                "source": "conversation",
        },

        )

        session.add(memory)
        session.commit()
        session.refresh(memory)

        return memory.id

def search_memories(
    user_id: str,
    query: str,
    top_k: int = 5,
    allowed_categories: tuple[str, ...] = DEFAULT_MEMORY_TYPES,
) -> list[dict]:
    """Retrieve relevant memories from approved categories."""

    if not user_id.strip():
        raise ValueError("user_id is required")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not allowed_categories:
        return []

    invalid_categories = (
        set(allowed_categories) - ALLOWED_MEMORY_TYPES
    )

    if invalid_categories:
        raise ValueError(
            f"Unsupported memory categories: {invalid_categories}"
        )

    query_embedding = get_embedding_model().encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    distance = ConversationMemory.embedding.cosine_distance(
        query_embedding
    )

    memory_category = ConversationMemory.metadata_["type"].astext

    statement = (
        select(ConversationMemory, distance.label("distance"))
        .where(
            ConversationMemory.user_id == user_id,
            memory_category.in_(allowed_categories),
        )
        .order_by(distance)
        .limit(top_k)
    )

    with SessionLocal() as session:
        results = session.execute(statement).all()

        return [
            {
                "id": memory.id,
                "memory": memory.memory_text,
                "session_id": memory.session_id,
                "category": (memory.metadata_ or {}).get("type"),
                "metadata": memory.metadata_ or {},
                "similarity": round(1 - distance_value, 4),
            }
            for memory, distance_value in results
            if 1 - distance_value >= RELEVANCE_THRESHOLD
        ]

def inspect_memories_by_ids(
    user_id: str,
    memory_ids: list[int],
) -> list[dict]:
    """Inspect selected memories belonging to one user."""

    if not user_id.strip():
        raise ValueError("user_id is required")

    if not memory_ids:
        return []

    statement = (
        select(ConversationMemory)
        .where(
            ConversationMemory.user_id == user_id,
            ConversationMemory.id.in_(memory_ids),
        )
        .order_by(ConversationMemory.id)
    )

    with SessionLocal() as session:
        records = session.execute(statement).scalars().all()

        return [
            {
                "id": memory.id,
                "memory": memory.memory_text,
                "metadata": memory.metadata_,
            }
            for memory in records
        ]
def delete_memories_by_ids(
    user_id: str,
    memory_ids: list[int],
) -> int:
    """Delete selected long-term memories belonging to one user."""

    if not user_id.strip():
        raise ValueError("user_id is required")

    if not memory_ids:
        return 0

    statement = (
        delete(ConversationMemory)
        .where(
            ConversationMemory.user_id == user_id,
            ConversationMemory.id.in_(memory_ids),
        )
    )

    with SessionLocal() as session:
        result = session.execute(statement)
        session.commit()

        return result.rowcount or 0
