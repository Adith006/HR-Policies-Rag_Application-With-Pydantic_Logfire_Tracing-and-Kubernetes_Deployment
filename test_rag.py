
from app.rag_pipeline import answer_question

question = (
    "What should an employee do if the first manager "
    "does not respond to a harassment complaint?"
)

result = answer_question(question)

print(f"Cache {result['cache']}: RAG pipeline executed.")

print("\nQuestion:")
print(result["question"])

print("\nAnswer:")
print(result["answer"])

print("\nSources:")
if result["sources"]:
    for source in result["sources"]:
        print(
            f"- {source['source']}, "
            f"page {source['page_number']}, "
            f"section: {source['section_title']}, "
            f"similarity: {source['similarity']}"
        )
else:
    print("No source citations available.")
