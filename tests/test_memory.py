
import json
import pytest

from app import memory


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def rpush(self, key, *values):
        self.operations.append(("rpush", key, values))
        return self

    def ltrim(self, key, start, end):
        self.operations.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self.operations.append(("expire", key, ttl))
        return self

    def execute(self):
        results = []

        for operation in self.operations:
            name = operation[0]

            if name == "rpush":
                _, key, values = operation
                self.client.data.setdefault(key, []).extend(values)
                results.append(len(self.client.data[key]))

            elif name == "ltrim":
                _, key, start, end = operation
                values = self.client.data.get(key, [])
                stop = end + 1 if end >= 0 else len(values) + end + 1
                self.client.data[key] = values[start:stop]
                results.append(True)

            elif name == "expire":
                _, key, ttl = operation
                self.client.expirations[key] = ttl
                results.append(True)

        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.expirations = {}

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def lrange(self, key, start, end):
        values = self.data.get(key, [])

        if end == -1:
            return values[start:]

        return values[start:end + 1]

    def delete(self, key):
        existed = key in self.data
        self.data.pop(key, None)
        self.expirations.pop(key, None)
        return int(existed)


@pytest.fixture
def conversation_memory(monkeypatch):
    fake_redis = FakeRedis()

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(
        memory.redis.Redis,
        "from_url",
        lambda *args, **kwargs: fake_redis,
    )

    return memory.ConversationMemory(), fake_redis


SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"


def test_add_turn_stores_user_and_assistant_messages(conversation_memory):
    store, client = conversation_memory

    store.add_turn(
        "user-1",
        SESSION_ID,
        "What is the leave policy?",
        "Employees should refer to the leave policy.",
    )

    history = store.get_history("user-1", SESSION_ID)

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is the leave policy?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == (
        "Employees should refer to the leave policy."
    )


def test_history_preserves_message_order(conversation_memory):
    store, _ = conversation_memory

    store.add_turn("user-1", SESSION_ID, "Question 1", "Answer 1")
    store.add_turn("user-1", SESSION_ID, "Question 2", "Answer 2")

    history = store.get_history("user-1", SESSION_ID)

    assert [message["content"] for message in history] == [
        "Question 1",
        "Answer 1",
        "Question 2",
        "Answer 2",
    ]


def test_history_is_limited_to_latest_twenty_messages(conversation_memory):
    store, _ = conversation_memory

    for index in range(12):
        store.add_turn(
            "user-1",
            SESSION_ID,
            f"Question {index}",
            f"Answer {index}",
        )

    history = store.get_history("user-1", SESSION_ID)

    assert len(history) == 20
    assert history[0]["content"] == "Question 2"
    assert history[-1]["content"] == "Answer 11"


def test_history_has_twenty_four_hour_ttl(conversation_memory):
    store, client = conversation_memory

    store.add_turn("user-1", SESSION_ID, "Question", "Answer")

    key = f"conversation:user-1:{SESSION_ID}"

    assert client.expirations[key] == 86400


def test_clear_history_deletes_conversation(conversation_memory):
    store, client = conversation_memory

    store.add_turn("user-1", SESSION_ID, "Question", "Answer")
    store.clear_history("user-1", SESSION_ID)

    assert store.get_history("user-1", SESSION_ID) == []
    assert f"conversation:user-1:{SESSION_ID}" not in client.data


def test_different_users_have_separate_histories(conversation_memory):
    store, _ = conversation_memory

    store.add_turn("user-1", SESSION_ID, "User 1 question", "User 1 answer")
    store.add_turn("user-2", SESSION_ID, "User 2 question", "User 2 answer")

    history_1 = store.get_history("user-1", SESSION_ID)
    history_2 = store.get_history("user-2", SESSION_ID)

    assert history_1[0]["content"] == "User 1 question"
    assert history_2[0]["content"] == "User 2 question"


@pytest.mark.parametrize(
    "session_id",
    ["", "invalid-session-id", "not-a-uuid"],
)
def test_invalid_session_id_is_rejected(conversation_memory, session_id):
    store, _ = conversation_memory

    with pytest.raises(ValueError):
        store.get_history("user-1", session_id)


def test_empty_user_id_is_rejected(conversation_memory):
    store, _ = conversation_memory

    with pytest.raises(ValueError, match="user ID"):
        store.get_history(" ", SESSION_ID)
