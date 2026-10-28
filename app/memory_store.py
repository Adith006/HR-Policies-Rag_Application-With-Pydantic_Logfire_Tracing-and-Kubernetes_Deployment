
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.db.models import ConversationMemory
from app.db.session import SessionLocal

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Similarity of 0.90 or higher is treated as a duplicate.
DUPLICATE_THRESHOLD = 0.90

# Return memories only when similarity meets this threshold.
RELEVANCE_THRESHOLD = 0.45


def store_memory(
    user_id: str,
    session_id: str,
    memory_text: str,
) -> int:
    """Store a memory unless a highly similar one already exists."""

    if not user_id.strip() or not session_id.strip():
        raise ValueError("user_id and session_id are required")

    memory_text = memory_text.strip()

    if not memory_text:
        raise ValueError("Memory text cannot be empty")

    embedding = model.encode(
        memory_text,
        normalize_embeddings=True,
    ).tolist()

    distance = ConversationMemory.embedding.cosine_distance(embedding)

    statement = (
        select(ConversationMemory, distance.label("distance"))
        .where(ConversationMemory.user_id == user_id)
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
            metadata_={"type": "conversation_memory"},
        )

        session.add(memory)
        session.commit()
        session.refresh(memory)

        return memory.id


def search_memories(
    user_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve relevant memories for one user."""

    if not user_id.strip():
        raise ValueError("user_id is required")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    distance = ConversationMemory.embedding.cosine_distance(
        query_embedding
    )

    statement = (
        select(ConversationMemory, distance.label("distance"))
        .where(ConversationMemory.user_id == user_id)
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
                "similarity": round(1 - distance_value, 4),
            }
            for memory, distance_value in results
            if 1 - distance_value >= RELEVANCE_THRESHOLD
        ]
