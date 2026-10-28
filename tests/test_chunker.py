import numpy as np
import pytest

from app.ingestion import chunker


class FakeEmbeddingModel:
    """Avoid loading the real embedding model during tests."""

    def encode(self, texts, normalize_embeddings=True):
        return np.tile(
            np.array([[1.0, 0.0]]),
            (len(texts), 1),
        )


@pytest.fixture(autouse=True)
def mock_embedding_model(monkeypatch):
    monkeypatch.setattr(
        chunker,
        "_get_model",
        lambda: FakeEmbeddingModel(),
    )


def test_empty_text_returns_no_chunks():
    assert chunker.chunk_text("   ") == []


@pytest.mark.parametrize("size", [0, -1])
def test_chunk_size_must_be_positive(size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker.chunk_text("Some text.", chunk_size=size)


@pytest.mark.parametrize("overlap", [-1, 10, 11])
def test_overlap_must_be_valid(overlap):
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunker.chunk_text(
            "Some text.",
            chunk_size=10,
            chunk_overlap=overlap,
        )


def test_paragraph_splitting_preserves_text_and_section():
    chunks = chunker.chunk_text(
        "First sentence. Second sentence.\n\nThird sentence.",
        chunk_size=100,
        chunk_overlap=0,
        initial_section="Employee Handbook",
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == (
        "First sentence. Second sentence. Third sentence."
    )
    assert chunks[0]["section_title"] == "Employee Handbook"


def test_chunks_do_not_mix_sections():
    lines = [
        {"text": "Leave Policy", "font_size": 12},
        {
            "text": "Employees should request leave in advance.",
            "font_size": 10,
        },
        {"text": "Reporting Policy", "font_size": 12},
        {
            "text": "Employees should report violations to HR.",
            "font_size": 10,
        },
    ]

    chunks = chunker.chunk_text(
        "Document content",
        chunk_size=200,
        chunk_overlap=0,
        lines=lines,
    )

    assert [c["section_title"] for c in chunks] == [
        "Leave Policy",
        "Reporting Policy",
    ]
    assert "Reporting Policy" not in chunks[0]["text"]


def test_bullet_is_not_treated_as_a_heading():
    lines = [
        {
            "text": "- Employees must follow this policy.",
            "font_size": 12,
        },
        {
            "text": "Managers should answer questions.",
            "font_size": 10,
        },
    ]

    chunks = chunker.chunk_text(
        "Document content",
        chunk_size=200,
        chunk_overlap=0,
        lines=lines,
        initial_section="General Policy",
    )

    assert chunks
    assert all(
        c["section_title"] == "General Policy"
        for c in chunks
    )


def test_long_sentence_is_split_within_chunk_size():
    text = " ".join(["word"] * 20) + "."

    chunks = chunker.chunk_text(
        text,
        chunk_size=20,
        chunk_overlap=0,
    )

    assert len(chunks) > 1
    assert all(
        0 < len(c["text"]) <= 20
        for c in chunks
    )
    assert " ".join(c["text"] for c in chunks) == text


def test_chunks_have_required_fields():
    chunks = chunker.chunk_text(
        "Employees should follow the handbook. Managers should help.",
        chunk_size=200,
        chunk_overlap=0,
        initial_section="Handbook",
    )

    assert chunks
    for chunk in chunks:
        assert isinstance(chunk["text"], str)
        assert chunk["text"].strip()
        assert chunk["section_title"] == "Handbook"