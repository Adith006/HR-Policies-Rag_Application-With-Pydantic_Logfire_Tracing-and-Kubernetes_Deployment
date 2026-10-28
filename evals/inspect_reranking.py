
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.search import semantic_search


DATASET_PATH = PROJECT_ROOT / "evals" / "dataset.json"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    print(f"Loading reranker: {RERANKER_NAME}")
    reranker = CrossEncoder(
        RERANKER_NAME,
        activation_fn=torch.nn.Sigmoid(),
    )

    for sample in samples:
        question = sample["question"]

        candidates = semantic_search(
            question=question,
            top_k=8,
            document_type="HR Policy",
            neighbor_window=1,
            rerank=False
        )

        print(f"\n{'=' * 80}")
        print(f"QUESTION: {question}")

        if not candidates:
            print("No candidates retrieved.")
            continue

        pairs = [
            (question, candidate["context_text"])
            for candidate in candidates
        ]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        for rerank_position, (candidate, score) in enumerate(
            ranked, start=1
        ):
            excerpt = " ".join(
                candidate["context_text"].split()
            )[:250]

            print(
                f"\nRerank: {rerank_position}"
                f" | Original similarity: {candidate['similarity']:.4f}"
                f" | Cross-encoder score: {float(score):.4f}"
                f" | Page: {candidate['page_number']}"
            )
            print(f"Context: {excerpt}")


if __name__ == "__main__":
    main()
