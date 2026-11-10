
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app import main


SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture
def client(monkeypatch):
    original_overrides = main.app.dependency_overrides.copy()

    main.app.dependency_overrides[
        main.verify_google_token
    ] = lambda: "test-user-123"

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()
    main.app.dependency_overrides.update(original_overrides)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_conversation(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "create_conversation",
        lambda **kwargs: {
            "id": kwargs["conversation_id"],
            "user_id": kwargs["user_id"],
        },
    )

    response = client.post(
        "/conversations",
        json={"conversation_id": SESSION_ID},
    )

    assert response.status_code == 200
    assert response.json()["id"] == SESSION_ID
    assert response.json()["user_id"] == "test-user-123"


def test_create_conversation_without_id(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "create_conversation",
        lambda **kwargs: {
            "id": "generated-id",
            "user_id": kwargs["user_id"],
        },
    )

    response = client.post("/conversations", json={})

    assert response.status_code == 200
    assert response.json()["user_id"] == "test-user-123"


def test_create_conversation_rejects_ownership_conflict(
    client, monkeypatch
):
    def raise_permission_error(**kwargs):
        raise PermissionError("Conversation belongs to another user.")

    monkeypatch.setattr(
        main,
        "create_conversation",
        raise_permission_error,
    )

    response = client.post(
        "/conversations",
        json={"conversation_id": SESSION_ID},
    )

    assert response.status_code == 403


def test_list_conversations(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_conversations",
        lambda **kwargs: [
            {"id": SESSION_ID, "title": "HR policy discussion"}
        ],
    )

    response = client.get("/conversations")

    assert response.status_code == 200
    assert len(response.json()["conversations"]) == 1
    assert response.json()["conversations"][0]["title"] == (
        "HR policy discussion"
    )


def test_retrieve_conversation_messages(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_messages",
        lambda **kwargs: [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    )

    response = client.get(
        f"/conversations/{SESSION_ID}/messages"
    )

    assert response.status_code == 200
    assert len(response.json()["messages"]) == 2


def test_retrieve_missing_conversation_returns_404(
    client, monkeypatch
):
    def raise_lookup_error(**kwargs):
        raise LookupError("Conversation not found.")

    monkeypatch.setattr(
        main,
        "get_messages",
        raise_lookup_error,
    )

    response = client.get(
        f"/conversations/{SESSION_ID}/messages"
    )

    assert response.status_code == 404


def test_pin_conversation(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "set_conversation_pinned",
        lambda **kwargs: True,
    )

    response = client.patch(
        f"/conversations/{SESSION_ID}/pin",
        json={"is_pinned": True},
    )

    assert response.status_code == 200
    assert response.json()["is_pinned"] is True


def test_unpin_conversation(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "set_conversation_pinned",
        lambda **kwargs: True,
    )

    response = client.patch(
        f"/conversations/{SESSION_ID}/pin",
        json={"is_pinned": False},
    )

    assert response.status_code == 200
    assert response.json()["is_pinned"] is False


def test_pin_missing_conversation_returns_404(
    client, monkeypatch
):
    monkeypatch.setattr(
        main,
        "set_conversation_pinned",
        lambda **kwargs: False,
    )

    response = client.patch(
        f"/conversations/{SESSION_ID}/pin",
        json={"is_pinned": True},
    )

    assert response.status_code == 404


def test_delete_conversation(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "delete_conversation",
        lambda **kwargs: True,
    )

    response = client.delete(
        f"/conversations/{SESSION_ID}"
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_delete_missing_conversation_returns_404(
    client, monkeypatch
):
    monkeypatch.setattr(
        main,
        "delete_conversation",
        lambda **kwargs: False,
    )

    response = client.delete(
        f"/conversations/{SESSION_ID}"
    )

    assert response.status_code == 404


def test_ask_question_persists_user_and_assistant_messages(
    client, monkeypatch
):
    saved_messages = []

    monkeypatch.setattr(
        main,
        "create_conversation",
        lambda **kwargs: {"id": SESSION_ID},
    )

    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: {
            "answer": "Employees should contact HR.",
            "sources": ["HR Handbook"],
            "retrieved_contexts": ["Contact HR for assistance."],
            "cache": False,
        },
    )

    monkeypatch.setattr(
        main,
        "save_message",
        lambda **kwargs: saved_messages.append(kwargs),
    )

    @contextmanager
    def fake_trace():
        yield [{"stage": "retrieval"}]

    monkeypatch.setattr(
        main,
        "capture_rag_trace",
        fake_trace,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Who should employees contact?",
            "session_id": SESSION_ID,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Employees should contact HR."
    )
    assert len(saved_messages) == 2
    assert saved_messages[0]["role"] == "user"
    assert saved_messages[1]["role"] == "assistant"
    assert saved_messages[1]["metadata"]["sources"] == [
        "HR Handbook"
    ]


def test_ask_explicit_memory_request(client, monkeypatch):
    saved_messages = []

    monkeypatch.setattr(
        main,
        "create_conversation",
        lambda **kwargs: {"id": SESSION_ID},
    )

    monkeypatch.setattr(
        main,
        "save_user_memories",
        lambda **kwargs: 2,
    )

    monkeypatch.setattr(
        main,
        "save_message",
        lambda **kwargs: saved_messages.append(kwargs),
    )

    response = client.post(
        "/ask",
        json={
            "question": "Remember that I prefer concise answers.",
            "session_id": SESSION_ID,
        },
    )

    assert response.status_code == 200
    assert response.json()["memory_saved"] is True
    assert response.json()["memory_count"] == 2
    assert len(saved_messages) == 2


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "session_id": SESSION_ID,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_session_id(client):
    response = client.post(
        "/ask",
        json={
            "question": "What is the leave policy?",
            "session_id": "invalid-session-id",
        },
    )

    assert response.status_code == 422


def test_unauthorized_request_is_rejected(client):
    main.app.dependency_overrides.pop(
        main.verify_google_token,
        None,
    )

    response = client.get("/conversations")

    assert response.status_code == 401


def test_ask_maps_value_error_to_400(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "create_conversation",
        lambda **kwargs: {"id": SESSION_ID},
    )

    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("Invalid question.")
        ),
    )

    @contextmanager
    def fake_trace():
        yield []

    monkeypatch.setattr(
        main,
        "capture_rag_trace",
        fake_trace,
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is the leave policy?",
            "session_id": SESSION_ID,
        },
    )

    assert response.status_code == 400
