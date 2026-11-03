import json
import hashlib
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import time
import logfire
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from langchain_core.embeddings import Embeddings
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
from app.embeddings import get_embedding_model


DATASET_PATH = PROJECT_ROOT / "evals" / "dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

PORTKEY_CONFIG = "pc-enterp-1a6936"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
EVALUATOR_VERSION = (
    "ragas-0.3.1-question-cards-f1-reference-similarity-v3"
)
def clean_score(value):
    """Convert a metric value into a valid rounded score."""

    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    return round(score, 4) if math.isfinite(score) else None


def calculate_cumulative_average(metric_scores: dict) -> float | None:
    """Calculate the mean of all four Ragas metrics."""

    scores = [
        metric_scores.get(metric)
        for metric in METRIC_NAMES
    ]

    if any(score is None for score in scores):
        return None

    return round(sum(scores) / len(METRIC_NAMES), 4)


def calculate_retrieval_f1(
    precision: float | None,
    recall: float | None,
) -> float | None:
    """Calculate the harmonic mean of context precision and recall."""

    if precision is None or recall is None:
        return None

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return round(
        2 * precision * recall / denominator,
        4,
    )


def calculate_answer_reference_similarity(
    generated_answer: str,
    reference_answer: str,
) -> float:
    """Calculate cosine similarity between generated and golden answers."""

    model = get_embedding_model()

    embeddings = model.encode(
        [generated_answer, reference_answer],
        normalize_embeddings=True,
    )

    similarity = float(embeddings[0] @ embeddings[1])

    return round(similarity, 4)


def load_dataset() -> list[dict]:
    """Load and validate the five golden evaluation samples."""

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list) or not samples:
        raise ValueError("The evaluation dataset is empty.")

    if len(samples) > 5:
        raise ValueError("The evaluation dataset cannot exceed five samples.")

    for index, sample in enumerate(samples):
        for field in ("question", "ground_truth"):
            value = sample.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Sample {index + 1} has an invalid {field}."
                )

        if sample["ground_truth"].startswith("REPLACE_WITH_"):
            raise ValueError(
                f"Golden answer {index + 1} has not been verified."
            )

    return samples



class SharedSentenceTransformerEmbeddings(Embeddings):
    """Reuse the RAG application's cached embedding model."""

    def __init__(self):
        self.model = get_embedding_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


def build_evaluator():
    """Configure the evaluation LLM and embedding model."""

    token = os.getenv("PORTKEY_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not token or not model:
        raise RuntimeError(
            "PORTKEY_API_KEY or GROQ_MODEL is missing."
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

    shared_embeddings = SharedSentenceTransformerEmbeddings()

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        shared_embeddings
    )

    return evaluator_llm, evaluator_embeddings

def collect_pipeline_result(
    question: str,
    include_details: bool = False,
):
    """Execute RAG and capture generated answer, retrieved contexts and sources."""

    captured = {}
    original_search = rag_pipeline.semantic_search

    def capture_search(**kwargs):
        results = original_search(**kwargs)
        captured["results"] = results
        return results

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
        # patch.object(rag_pipeline, "save_long_term_memories"),
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

    raw_results = captured.get("results", [])

    retrieved_items = [
        (item, item["context_text"].strip())
        for item in raw_results
        if isinstance(item.get("context_text"), str)
        and item["context_text"].strip()
    ]

    contexts = [context for _, context in retrieved_items]

    if not contexts:
        raise RuntimeError(
            "No document contexts retrieved for an evaluation question: "
            f"{question}"
        )

    answer = result.get("answer") if isinstance(result, dict) else None

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The RAG pipeline returned no usable answer.")

    sources = []

    for item, _ in retrieved_items:
        metadata = item.get("metadata") or {}

        sources.append({
            "source": item.get("source", "Unknown source"),
            "page_number": item.get("page_number"),
            "section_title": metadata.get(
                "section_title",
                "Unknown section",
            ),
            "similarity": item.get("similarity"),
        })

    if include_details:
        return answer, contexts, sources

    # Preserve compatibility with your existing diagnostic scripts.
    return answer, contexts




def run_evaluation(question_count: int = 5) -> dict:
    """Evaluate selected golden samples and generate question-wise reports."""

    if isinstance(question_count, bool) or not isinstance(question_count, int):
        raise ValueError("Question count must be an integer.")

    if not 1 <= question_count <= 5:
        raise ValueError("Question count must be between 1 and 5.")

    logfire.configure(send_to_logfire="if-token-present")

    samples = load_dataset()[:question_count]

    if len(samples) < question_count:
        raise ValueError("Not enough golden samples in the dataset.")
    total_start = time.perf_counter()

    # 1. Initialize evaluator
    stage_start = time.perf_counter()

    evaluator_llm, evaluator_embeddings = build_evaluator()

    logfire.info(
        "Evaluator initialization completed",
        elapsed_seconds=round(time.perf_counter() - stage_start, 2),
    )

    evaluation_rows = []
    collected_details = []

    # 2. Execute RAG pipeline
    stage_start = time.perf_counter()

    for index, sample in enumerate(samples, start=1):

        question_start = time.perf_counter()

        with logfire.span(
            "ragas.collect_sample",
            sample_number=index,
        ):
            answer, contexts, sources = collect_pipeline_result(
                sample["question"],
                include_details=True,
            )

        evaluation_rows.append({
            "user_input": sample["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": sample["ground_truth"],
        })

        collected_details.append({
            "question": sample["question"],
            "generated_answer": answer,
            "reference_answer": sample["ground_truth"],
            "retrieved_contexts": contexts,
            "sources": sources,
        })

        logfire.info(
            "RAG question completed",
            question_number=index,
            elapsed_seconds=round(
                time.perf_counter() - question_start, 2
            ),
        )

    logfire.info(
        "All RAG questions completed",
        elapsed_seconds=round(time.perf_counter() - stage_start, 2),
    )

    # 3. Run RAGAS metrics
    dataset = EvaluationDataset.from_list(evaluation_rows)

    stage_start = time.perf_counter()

    with logfire.span(
        "ragas.evaluate",
        sample_count=len(evaluation_rows),
        metric_count=len(METRIC_NAMES),
    ):
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

    logfire.info(
        "RAGAS metric evaluation completed",
        elapsed_seconds=round(time.perf_counter() - stage_start, 2),
    )

    logfire.info(
        "Total evaluation collection and metric time",
        elapsed_seconds=round(time.perf_counter() - total_start, 2),
    )


    scores = result.to_pandas()

    question_scores = []

    # 3. Calculate metrics independently for every question.
    for index, sample in enumerate(samples):

        row = scores.iloc[index]

        individual_scores = {
            metric: clean_score(row.get(metric))
            for metric in METRIC_NAMES
        }

        cumulative_average = calculate_cumulative_average(
            individual_scores
        )

        retrieval_f1 = calculate_retrieval_f1(
            individual_scores["context_precision"],
            individual_scores["context_recall"],
        )

        details = collected_details[index]

        answer_similarity = calculate_answer_reference_similarity(
            details["generated_answer"],
            details["reference_answer"],
        )

        question_scores.append({
            "question_number": index + 1,
            "question": details["question"],
            "generated_answer": details["generated_answer"],
            "reference_answer": details["reference_answer"],
            "scores": individual_scores,
            "cumulative_average": cumulative_average,
            "retrieval_f1": retrieval_f1,
            "answer_reference_similarity": answer_similarity,
            "retrieved_contexts": details["retrieved_contexts"],
            "sources": details["sources"],
        })

    # 4. Calculate averages for the selected questions only.
    metric_averages = {}

    for metric in METRIC_NAMES:
        valid_scores = [
            item["scores"][metric]
            for item in question_scores
            if item["scores"][metric] is not None
        ]

        metric_averages[metric] = (
            round(sum(valid_scores) / len(valid_scores), 4)
            if valid_scores
            else None
        )

    valid_question_averages = [
        item["cumulative_average"]
        for item in question_scores
        if item["cumulative_average"] is not None
    ]

    overall_cumulative_average = (
        round(
            sum(valid_question_averages) /
            len(valid_question_averages),
            4,
        )
        if valid_question_averages
        else None
    )

    # 5. Save evaluation results.
    report = {
        "question_count": len(question_scores),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metric_averages": metric_averages,
        "cumulative_average": overall_cumulative_average,
        "question_scores": question_scores,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores.insert(
        0,
        "question",
        [sample["question"] for sample in samples],
    )

    scores["cumulative_average"] = [
        item["cumulative_average"]
        for item in question_scores
    ]

    scores["retrieval_f1"] = [
        item["retrieval_f1"]
        for item in question_scores
    ]

    scores["answer_reference_similarity"] = [
        item["answer_reference_similarity"]
        for item in question_scores
    ]

    scores.to_csv(
        RESULTS_DIR / "latest_ragas_results.csv",
        index=False,
        encoding="utf-8",
    )

    return report


def main():
    report = run_evaluation(5)

    print("\nIndividual metric averages:")

    for metric, score in report["metric_averages"].items():
        print(f"{metric}: {score}")

    print(
        "\nCumulative average:",
        report["cumulative_average"],
    )


if __name__ == "__main__":
    main()
