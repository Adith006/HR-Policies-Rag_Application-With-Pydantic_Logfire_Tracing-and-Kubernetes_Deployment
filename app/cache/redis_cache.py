
import json
import os

import redis

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)

CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))


def get_cached_answer(cache_key: str) -> dict | None:
    """Return a cached RAG result, or None if invalid."""
    cached = redis_client.get(cache_key)

    if not cached:
        return None

    try:
        data = json.loads(cached)
    except json.JSONDecodeError:
        redis_client.delete(cache_key)
        return None

    required_fields = {"answer", "sources", "context"}

    if (
        not isinstance(data, dict)
        or not required_fields.issubset(data)
        or not isinstance(data["answer"], str)
        or not isinstance(data["sources"], list)
        or not isinstance(data["context"], str)
    ):
        redis_client.delete(cache_key)
        return None

    return data


def cache_answer(
    cache_key: str,
    answer: str,
    sources: list[dict],
    context: str,
) -> None:
    """Cache the answer, source citations, and grounding context."""
    payload = {
        "answer": answer,
        "sources": sources,
        "context": context,
    }

    redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(payload),
    )
