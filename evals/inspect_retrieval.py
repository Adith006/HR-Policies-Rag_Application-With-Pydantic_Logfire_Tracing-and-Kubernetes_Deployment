
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.search import semantic_search


def main():
    dataset_path = PROJECT_ROOT / "evals" / "dataset.json"
    with dataset_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    for sample in samples:
        # if "vacation entitlement" not in sample["question"].lower():
        #     continue
        question = sample["question"]

        results = semantic_search(
            question=question,
            top_k=3,
            document_type="HR Policy",
            neighbor_window=1,
        )

        print(f"\n{'=' * 80}")
        print(f"QUESTION: {question}")

        if not results:
            print("No results retrieved.")
            continue

        for rank, result in enumerate(results, start=1):
            excerpt = " ".join(
                result["context_text"].split()
            )[:250]

            print(
                f"\nRank: {rank}"
                f" | Similarity: {result['similarity']:.4f}"
                f" | Rerank score: {result.get('rerank_score', 'MISSING')}"
                f" | Page: {result['page_number']}"
            )
            print(f"Context excerpt: {excerpt}")


if __name__ == "__main__":
    main()
