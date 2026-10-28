
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import logfire
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

import app.rag_pipeline as rag_pipeline


DATASET_PATH = PROJECT_ROOT / "evals" / "dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
PORTKEY_CONFIG = "pc-enterp-1a6936"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_dataset() -> list[dict]:
    """Load and validate manually reviewed evaluation examples."""

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list) or not samples:
        raise ValueError("The evaluation dataset must be a non-empty list.")

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Dataset item {index} must be an object.")

        for field in ("question", "ground_truth"):
            value = sample.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Dataset item {index} requires a non-empty '{field}'."
                )

        if sample["ground_truth"].startswith("REPLACE_WITH_"):
            raise ValueError(
                f"Add a verified reference answer to dataset item {index}."
            )

    return samples


def build_evaluator():
    """Configure the evaluation LLM and local embedding model."""

    token = os.getenv("PORTKEY_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not token or not model:
        raise RuntimeError(
            "PORTKEY_API_KEY or GROQ_MODEL is missing from .env."
        )

    chat_model = ChatOpenAI(
        model=model,
        api_key=token,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=createHeaders(
            api_key=token,
            config=PORTKEY_CONFIG,
        ),
        temperature=0,
        max_tokens=1024,
    )

    evaluator_llm = LangchainLLMWrapper(chat_model)

    local_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    return evaluator_llm, evaluator_embeddings


def collect_pipeline_result(question: str) -> tuple[str, list[str]]:
    """Run the RAG pipeline while avoiding evaluation-time state writes."""

    captured: dict = {}
    original_search = rag_pipeline.semantic_search

    def capture_search(**kwargs):
        results = original_search(**kwargs)
        captured["results"] = results
        return results

    # Isolate the evaluation from existing conversation state and cache.
    # Keep the real validation, retrieval, generation, and grounding logic.
    with (
        patch.object(
            rag_pipeline.conversation_memory,
            "get_history",
            return_value=[],
        ),
        patch.object(
            rag_pipeline.conversation_memory,
            "add_turn",
        ),
        patch.object(
            rag_pipeline,
            "search_memories",
            return_value=[],
        ),
        patch.object(
            rag_pipeline,
            "get_cached_answer",
            return_value=None,
        ),
        patch.object(rag_pipeline, "cache_answer"),
        patch.object(rag_pipeline, "save_long_term_memories"),
        patch.object(
            rag_pipeline,
            "semantic_search",
            side_effect=capture_search,
        ),
    ):
        result = rag_pipeline.answer_question(
            question=question,
            session_id=str(uuid4()),
            user_id=f"ragas-evaluation-{uuid4().hex}",
        )

    results = captured.get("results", [])
    contexts = [
        item["context_text"].strip()
        for item in results
        if isinstance(item.get("context_text"), str)
        and item["context_text"].strip()
    ]

    if not contexts:
        raise RuntimeError(
            "No document contexts were retrieved for an evaluation question: "
            f"{question}"
        )

    answer = result.get("answer") if isinstance(result, dict) else None

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The RAG pipeline returned no usable answer.")

    return answer, contexts


def main() -> None:
    logfire.configure(send_to_logfire="if-token-present")

    samples = load_dataset()
    evaluator_llm, evaluator_embeddings = build_evaluator()

    evaluation_rows = []

    for index, sample in enumerate(samples, start=1):
        # Avoid recording actual questions and answers as span attributes.
        with logfire.span("ragas.collect_sample", sample_number=index):
            answer, contexts = collect_pipeline_result(sample["question"])

            evaluation_rows.append(
                {
                    "user_input": sample["question"],
                    "retrieved_contexts": contexts,
                    "response": answer,
                    "reference": sample["ground_truth"],
                }
            )

        print(f"Collected evaluation sample {index}/{len(samples)}")

    dataset = EvaluationDataset.from_list(evaluation_rows)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    with logfire.span(
        "ragas.evaluate",
        sample_count=len(evaluation_rows),
        metric_count=len(metrics),
    ):
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

    scores = results.to_pandas()
    scores.insert(
        0,
        "question",
        [row["user_input"] for row in evaluation_rows],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "latest_ragas_results.csv"
    scores.to_csv(report_path, index=False, encoding="utf-8")

    print("\nRagas evaluation results:")
    print(scores.to_string(index=False))
    print(f"\nDetailed results saved to: {report_path}")


if __name__ == "__main__":
    main()
