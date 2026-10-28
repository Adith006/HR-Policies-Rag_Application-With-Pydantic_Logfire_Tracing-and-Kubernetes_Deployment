
from pathlib import Path

from app.ingestion.pdf_reader import extract_pdf
from app.ingestion.chunker import chunk_text


def detect_document_type(file_name: str) -> str:
    name = file_name.lower()

    if "manual" in name or "policy" in name:
        return "HR Policy"

    return "General Document"


def process_pdf(file_path: str) -> list[dict]:
    pdf_path = Path(file_path)
    pages = extract_pdf(str(pdf_path))

    documents = []
    document_type = detect_document_type(pdf_path.name)
    current_section = "Unknown"

    for page in pages:
        chunks = chunk_text(
            page["text"],
            lines=page["lines"],
            initial_section=current_section,
        )

        for chunk_index, chunk in enumerate(chunks):
            documents.append({
                "source": pdf_path.name,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk["text"],
                "metadata": {
                    "document_type": document_type,
                    "section_title": chunk["section_title"],
                },
            })

        # Carry the last detected section into the next page.
        if chunks:
            current_section = chunks[-1]["section_title"]

    return documents


if __name__ == "__main__":
    pdf_path = Path("sample_documents/manual.pdf")
    chunks = process_pdf(str(pdf_path))

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:10]:
        print(f"\nPage: {chunk['page_number']}")
        print(f"Chunk: {chunk['chunk_index']}")
        print(f"Metadata: {chunk['metadata']}")
        print(chunk["text"][:250])
