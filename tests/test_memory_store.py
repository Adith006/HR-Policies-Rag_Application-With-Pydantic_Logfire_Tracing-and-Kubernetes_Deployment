
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app import memory_store
from types import SimpleNamespace


class FakeEmbeddingModel:
    def encode(self, text, normalize_embeddings=True):
        return np.array([1.0] + [0.0] * 383)


class FakeSession:
    def __init__(self, execute_result=None):
        self.execute_result = execute_result
        self.added = []
        self.committed = False
        self.refreshed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement):
        return self.execute_result

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        obj.id = 101
        self.refreshed.append(obj)

@pytest.fixture
def fake_session(monkeypatch):
    result = SimpleNamespace(first=lambda: None)
    session = FakeSession(execute_result=result)

    monkeypatch.setattr(
        memory_store,
        "SessionLocal",
        lambda: session,
    )
    monkeypatch.setattr(
        memory_store,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    return session



def test_store_memory_saves_categorized_memory(fake_session):
    memory_id = memory_store.store_memory(
        user_id="user-1",
        session_id="123e4567-e89b-12d3-a456-426614174000",
        memory_text="Prefers concise answers",
        memory_type="user_preference",
    )

    assert memory_id == 101
    assert len(fake_session.added) == 1
    assert fake_session.committed

    saved = fake_session.added[0]
    assert saved.user_id == "user-1"
    assert saved.memory_text == "Prefers concise answers"
    assert saved.metadata_["type"] == "user_preference"
    assert saved.metadata_["source"] == "conversation"


def test_store_memory_protects_category_and_source_metadata(fake_session):
    memory_store.store_memory(
        user_id="user-1",
        session_id="123e4567-e89b-12d3-a456-426614174000",
        memory_text="Uses Python for AI projects",
        memory_type="project_context",
        metadata={
            "type": "user_preference",
            "source": "untrusted",
        },
    )

    saved = fake_session.added[0]

    assert saved.metadata_["type"] == "project_context"
    assert saved.metadata_["source"] == "conversation"


@pytest.mark.parametrize(
    "text",
    [
        "postgresql://user:password@example.com/database",
        "api_key=secret-value",
        "access_token: secret-value",
        "password=my-secret",
        "-----BEGIN RSA PRIVATE KEY-----",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_sensitive_information_is_detected(text):
    assert memory_store.contains_sensitive_data(text)


def test_store_memory_rejects_sensitive_information(fake_session):
    with pytest.raises(ValueError, match="sensitive information"):
        memory_store.store_memory(
            user_id="user-1",
            session_id="123e4567-e89b-12d3-a456-426614174000",
            memory_text="password=secret123",
            memory_type="user_preference",
        )

    assert fake_session.added == []
    assert not fake_session.committed


def test_store_memory_rejects_invalid_memory_type(fake_session):
    with pytest.raises(ValueError, match="Invalid memory type"):
        memory_store.store_memory(
            user_id="user-1",
            session_id="123e4567-e89b-12d3-a456-426614174000",
            memory_text="Some memory",
            memory_type="invalid_type",
        )


def test_store_memory_rejects_empty_text(fake_session):
    with pytest.raises(ValueError, match="cannot be empty"):
        memory_store.store_memory(
            user_id="user-1",
            session_id="123e4567-e89b-12d3-a456-426614174000",
            memory_text="   ",
            memory_type="user_preference",
        )


def test_duplicate_memory_is_not_inserted(monkeypatch):
    existing = SimpleNamespace(id=55)
    result = MagicMock()
    result.first.return_value = (existing, 0.05)

    session = FakeSession(execute_result=result)

    monkeypatch.setattr(
        memory_store,
        "SessionLocal",
        lambda: session,
    )
    monkeypatch.setattr(
        memory_store,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    memory_id = memory_store.store_memory(
        user_id="user-1",
        session_id="123e4567-e89b-12d3-a456-426614174000",
        memory_text="Prefers concise answers",
        memory_type="user_preference",
    )

    assert memory_id == 55
    assert session.added == []
    assert not session.committed


def test_search_memories_returns_relevant_results(monkeypatch):
    records = [
        (
            SimpleNamespace(
                id=1,
                memory_text="Prefers concise answers",
                session_id="session-1",
                metadata_={"type": "user_preference"},
            ),
            0.10,
        ),
        (
            SimpleNamespace(
                id=2,
                memory_text="Uses Python",
                session_id="session-2",
                metadata_={"type": "user_provided_fact"},
            ),
            0.80,
        ),
    ]

    result = MagicMock()
    result.all.return_value = records

    session = FakeSession(execute_result=result)

    monkeypatch.setattr(
        memory_store,
        "SessionLocal",
        lambda: session,
    )
    monkeypatch.setattr(
        memory_store,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    memories = memory_store.search_memories(
        user_id="user-1",
        query="What are my preferences?",
    )

    assert len(memories) == 1
    assert memories[0]["id"] == 1
    assert memories[0]["memory"] == "Prefers concise answers"
    assert memories[0]["similarity"] == 0.9


def test_search_memories_rejects_invalid_categories(fake_session):
    with pytest.raises(ValueError, match="Unsupported memory categories"):
        memory_store.search_memories(
            user_id="user-1",
            query="Question",
            allowed_categories=("invalid_category",),
        )


def test_search_memories_returns_empty_for_no_categories(fake_session):
    memories = memory_store.search_memories(
        user_id="user-1",
        query="Question",
        allowed_categories=(),
    )

    assert memories == []


def test_inspect_memories_returns_selected_records(monkeypatch):
    records = [
        SimpleNamespace(
            id=10,
            memory_text="Uses Python",
            metadata_={"type": "user_provided_fact"},
        )
    ]

    scalars = MagicMock()
    scalars.all.return_value = records

    result = MagicMock()
    result.scalars.return_value = scalars

    session = FakeSession(execute_result=result)

    monkeypatch.setattr(
        memory_store,
        "SessionLocal",
        lambda: session,
    )

    memories = memory_store.inspect_memories_by_ids(
        "user-1",
        [10],
    )

    assert len(memories) == 1
    assert memories[0]["id"] == 10
    assert memories[0]["memory"] == "Uses Python"


def test_delete_memories_returns_deleted_count(monkeypatch):
    result = SimpleNamespace(rowcount=2)
    session = FakeSession(execute_result=result)

    monkeypatch.setattr(
        memory_store,
        "SessionLocal",
        lambda: session,
    )

    deleted = memory_store.delete_memories_by_ids(
        "user-1",
        [10, 11],
    )

    assert deleted == 2
    assert session.committed


def test_delete_memories_returns_zero_for_empty_ids(fake_session):
    deleted = memory_store.delete_memories_by_ids(
        "user-1",
        [],
    )

    assert deleted == 0
    assert not fake_session.committed
