from app.ingestion.pipeline import process_pdf

chunks = process_pdf("sample_documents/manual.pdf")

for chunk in chunks:
    if chunk["page_number"] == 3:
        print(
            f"Chunk {chunk['chunk_index']} | "
            f"Section: {chunk['metadata']['section_title']} | "
            f"Text: {chunk['text'][:100]}"
        )
        print("-" * 60)