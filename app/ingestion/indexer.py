
from app.embeddings import get_embedding_model
from sqlalchemy import delete

from app.db.session import SessionLocal
from app.db.models import DocumentChunk
from app.ingestion.pipeline import process_pdf


def build_embedding_text(chunk: dict) -> str:
    """Combine section context and chunk content for embedding."""
    metadata = chunk.get("metadata", {})
    section_title = metadata.get("section_title", "General")

    return (
        f"Section: {section_title}\n"
        f"Content: {chunk['text']}"
    )


def index_pdf(file_path: str) -> int:
    chunks = process_pdf(file_path)

    if not chunks:
        print("No text chunks found. Existing records were not changed.")
        return 0

    model = get_embedding_model()

    texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    records = [
        DocumentChunk(
            source=chunk["source"],
            page_number=chunk["page_number"],
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            embedding=embedding.tolist(),
            metadata_=chunk.get("metadata", {}),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    source_name = chunks[0]["source"]

    with SessionLocal() as session:
        try:
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.source == source_name
                )
            )

            session.add_all(records)
            session.commit()

        except Exception:
            session.rollback()
            raise

    return len(records)


if __name__ == "__main__":
    pdf_path = "sample_documents/manual.pdf"
    count = index_pdf(pdf_path)
    print(f"Indexed {count} chunks successfully.")
