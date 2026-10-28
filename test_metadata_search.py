
from app.retrieval.search import semantic_search

results = semantic_search(
    "What is the equal employment policy?",
    top_k=3,
    document_type="HR Policy",
)

for result in results:
    print("Source:", result["source"])
    print("Page:", result["page_number"])
    print("Similarity:", result["similarity"])
    print("Metadata:", result["metadata"])
    print("Text:", result["text"][:200])
    print("Context chunk indices:", result["context_chunk_indices"])
    print("Expanded context:", result["context_text"][:600])
    print("Chunk index:", result["chunk_index"])
    print("-" * 40)
