
import json
import os
from uuid import UUID

import redis


class ConversationMemory:
    def __init__(self):
        self.client = redis.Redis.from_url(
            os.environ["REDIS_URL"],
            decode_responses=True,
        )
        self.ttl = 86400  # 24 hours
        self.max_messages = 20

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        return str(UUID(session_id))

    def _key(self, session_id: str) -> str:
        session_id = self._validate_session_id(session_id)
        return f"conversation:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        messages = self.client.lrange(self._key(session_id), 0, -1)
        return [json.loads(message) for message in messages]

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        key = self._key(session_id)

        with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(
                key,
                json.dumps({"role": "user", "content": question}),
                json.dumps({"role": "assistant", "content": answer}),
            )
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, self.ttl)
            pipe.execute()

    def clear_history(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))
