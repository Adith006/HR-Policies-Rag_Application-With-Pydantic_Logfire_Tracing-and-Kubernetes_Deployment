
from app.retrieval.search import semantic_search

question = "What should an employee do to report workplace harassment?"

results = semantic_search(
    question=question,
    top_k=3,
    document_type="HR Policy",
    neighbor_window=1,
)

print(f"Results found: {len(results)}")

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page_number']}")
    print(f"Section: {result['metadata']['section_title']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Text:\n{result['context_text']}")
    print("-" * 60)
