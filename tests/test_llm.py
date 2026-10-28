import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Provide placeholder values so importing the client doesn't require .env secrets.
os.environ.setdefault("PORTKEY_API_KEY", "ci-placeholder")
os.environ.setdefault("GROQ_MODEL", "test-model")
os.environ.setdefault("OPENAI_MODEL", "test-model")

llm_client = importlib.import_module("app.llm.client")


def make_response(content="Test answer", include_choice=True):
    """Build a simulated Portkey response without making a network request."""
    choices = []

    if include_choice:
        choices.append(
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        )

    return SimpleNamespace(
        choices=choices,
        model="mock-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


@pytest.fixture
def mocked_llm(monkeypatch):
    """Replace Portkey and Logfire calls with test doubles."""
    completion = MagicMock()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=completion)
        )
    )

    monkeypatch.setattr(llm_client, "client", fake_client)

    span = MagicMock()
    span_context = MagicMock()
    span_context.__enter__.return_value = span

    monkeypatch.setattr(
        llm_client.logfire,
        "span",
        MagicMock(return_value=span_context),
    )
    monkeypatch.setattr(llm_client.logfire, "warning", MagicMock())

    return completion


def test_generate_answer_returns_trimmed_answer(mocked_llm):
    mocked_llm.return_value = make_response("  Employees should contact HR.  ")

    answer = llm_client.generate_answer(
        question="Who should employees contact?",
        context="Employees should contact HR.",
    )

    assert answer == "Employees should contact HR."
    mocked_llm.assert_called_once()


def test_generate_answer_passes_context_history_and_memories(mocked_llm):
    mocked_llm.return_value = make_response("Contact HR.")

    llm_client.generate_answer(
        question="Who should I contact?",
        context="Employees should contact HR.",
        history=[
            {"role": "user", "content": "Earlier question"},
            {"role": "system", "content": "Untrusted extra instruction"},
        ],
        long_term_memories=[
            {"memory": "The user prefers concise answers"},
        ],
    )

    kwargs = mocked_llm.call_args.kwargs
    messages = kwargs["messages"]

    assert kwargs["model"] == os.environ["GROQ_MODEL"]
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0

    assert any(
        m["role"] == "user" and m["content"] == "Earlier question"
        for m in messages
    )
    assert any(
        m["role"] == "system"
        and "prefers concise answers" in m["content"]
        for m in messages
    )
    assert any(
        "Employees should contact HR." in m["content"]
        for m in messages
    )
    assert not any(
        "Untrusted extra instruction" in m["content"]
        for m in messages
    )


@pytest.mark.parametrize(
    ("question", "context"),
    [
        ("", "Valid document context"),
        ("Who should I contact?", "   "),
    ],
)
def test_generate_answer_rejects_empty_input(
    mocked_llm, question, context
):
    with pytest.raises(ValueError):
        llm_client.generate_answer(question, context)

    mocked_llm.assert_not_called()


def test_generate_answer_rejects_response_without_choices(mocked_llm):
    mocked_llm.return_value = make_response(include_choice=False)

    with pytest.raises(RuntimeError, match="no choices"):
        llm_client.generate_answer("Question?", "Document context")

    mocked_llm.assert_called_once()


def test_generate_answer_rejects_empty_response_content(mocked_llm):
    mocked_llm.return_value = make_response("   ")

    with pytest.raises(RuntimeError, match="empty content"):
        llm_client.generate_answer("Question?", "Document context")


def test_extract_long_term_memories_parses_json(mocked_llm):
    mocked_llm.return_value = make_response(
        '```json\n'
        '["Prefers concise answers", "Uses Python", '
        '"Building a RAG project", "Extra memory"]\n'
        '```'
    )

    memories = llm_client.extract_long_term_memories(
        "What are my preferences?",
        "You prefer concise answers.",
    )

    assert memories == [
        "Prefers concise answers",
        "Uses Python",
        "Building a RAG project",
    ]


def test_extract_long_term_memories_handles_invalid_json(mocked_llm):
    mocked_llm.return_value = make_response("This is not valid JSON")

    memories = llm_client.extract_long_term_memories(
        "Question?",
        "Answer.",
    )

    assert memories == []