
import json
import os

import redis

import os
from dotenv import load_dotenv

load_dotenv()


REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not configured.")

redis_client = redis.from_url(
    REDIS_URL,
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
    retrieved_contexts: list[dict] | None = None,
) -> None:
    """Cache the answer, source citations, and retrieved context chunks."""
    payload = {
        "answer": answer,
        "sources": sources,
        "context": context,
        "retrieved_contexts": retrieved_contexts or [],
    }

    redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(payload),
    )