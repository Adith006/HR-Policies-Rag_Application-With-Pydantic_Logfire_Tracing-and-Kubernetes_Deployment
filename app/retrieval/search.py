
import logfire
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.db.models import DocumentChunk
from app.db.session import SessionLocal

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_SCORE_MARGIN = 0.15

model = SentenceTransformer(MODEL_NAME)
_reranker = None


def _get_reranker():
    """Load the cross-encoder once, when reranking is first required."""
    global _reranker

    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(
            RERANKER_NAME,
            activation_fn=torch.nn.Sigmoid(),
        )

    return _reranker


def _build_context(session, chunk, neighbor_window: int) -> tuple[str, list[int]]:
    """Build context for a selected chunk using adjacent chunks."""
    context_indices = [chunk.chunk_index]

    if neighbor_window > 0:
        context_indices = list(
            range(
                max(0, chunk.chunk_index - neighbor_window),
                chunk.chunk_index + neighbor_window + 1,
            )
        )

    with logfire.span("retrieval.neighbor_query") as span:
        neighbor_statement = (
            select(DocumentChunk)
            .where(
                DocumentChunk.source == chunk.source,
                DocumentChunk.page_number == chunk.page_number,
                DocumentChunk.chunk_index.in_(context_indices),
                DocumentChunk.metadata_["section_title"].astext
                == chunk.metadata_["section_title"],
            )
            .order_by(DocumentChunk.chunk_index)
        )

        neighbors = session.execute(
            neighbor_statement
        ).scalars().all()

        span.set_attribute(
            "retrieval.neighbor_count",
            len(neighbors),
        )

    with logfire.span("retrieval.context_assembly") as span:
        context_parts = []

        for neighbor in neighbors:
            text = neighbor.text.strip()

            if context_parts:
                previous_text = context_parts[-1]
                overlap_length = min(
                    len(previous_text),
                    len(text),
                )

                while overlap_length > 0:
                    if previous_text.endswith(
                        text[:overlap_length]
                    ):
                        break
                    overlap_length -= 1

                if overlap_length > 0:
                    text = text[overlap_length:].lstrip()

            if text:
                context_parts.append(text)

        context_text = "\n\n".join(context_parts)

        span.set_attribute(
            "retrieval.context_char_count",
            len(context_text),
        )
        span.set_attribute(
            "retrieval.context_chunk_count",
            len(context_parts),
        )

    return context_text, [neighbor.chunk_index for neighbor in neighbors]


def semantic_search(
    question: str,
    top_k: int = 5,
    document_type: str | None = None,
    neighbor_window: int = 0,
    rerank: bool = True,
) -> list[dict]:
    """Retrieve vector candidates, rerank them, and expand selected context."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if neighbor_window < 0:
        raise ValueError("neighbor_window cannot be negative")

    candidate_limit = top_k * 3 if rerank else top_k

    # 1. Generate the query embedding.
    with logfire.span("retrieval.query_embedding") as span:
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        span.set_attribute(
            "embedding.dimensions",
            len(query_embedding),
        )

    # 2. Retrieve a wider candidate pool using pgvector.
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    statement = select(
        DocumentChunk,
        distance.label("distance"),
    )

    if document_type:
        statement = statement.where(
            DocumentChunk.metadata_["document_type"].astext
            == document_type
        )

    statement = statement.order_by(distance).limit(candidate_limit)

    candidates = []

    with SessionLocal() as session:
        with logfire.span("retrieval.pgvector_query") as span:
            matches = session.execute(statement).all()

            span.set_attribute(
                "retrieval.candidate_count",
                len(matches),
            )
            span.set_attribute(
                "retrieval.requested_top_k",
                top_k,
            )
            span.set_attribute(
                "retrieval.candidate_limit",
                candidate_limit,
            )

        for query_rank, (chunk, distance_value) in enumerate(matches, start=1):
            # Avoid duplicate chunks before reranking.
            if any(
                item["_chunk"].source == chunk.source
                and item["_chunk"].page_number == chunk.page_number
                and item["_chunk"].chunk_index == chunk.chunk_index
                for item in candidates
            ):
                continue

            if any(item["text"] == chunk.text for item in candidates):
                continue

            candidates.append(
                {
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata_,
                    "similarity": round(
                        1 - float(distance_value),
                        4,
                    ),
                    "context_text": chunk.text.strip(),
                    "context_chunk_indices": [chunk.chunk_index],
                    "_chunk": chunk,
                    "_query_rank": query_rank,
                }
            )

    # 3. Rerank the candidates using question-passage relevance.
    if rerank and candidates:
        with logfire.span("retrieval.cross_encoder_rerank") as span:
            span.set_attribute(
                "reranker.model",
                RERANKER_NAME,
            )
            span.set_attribute(
                "reranker.candidate_count",
                len(candidates),
            )

            reranker = _get_reranker()

            pairs = [
                (question, candidate["text"])
                for candidate in candidates
            ]

            scores = reranker.predict(
                pairs,
                show_progress_bar=False,
            )

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            candidates.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

            span.set_attribute(
                "reranker.top_score",
                candidates[0]["rerank_score"],
            )

    # 4. Log ranked candidate diagnostics without recording document text.
    for rerank_rank, candidate in enumerate(candidates, start=1):
        with logfire.span("retrieval.candidate_diagnostic") as span:
            span.set_attribute(
                "retrieval.query_rank",
                candidate["_query_rank"],
            )
            span.set_attribute(
                "retrieval.rerank_rank",
                rerank_rank,
            )
            span.set_attribute(
                "retrieval.similarity",
                candidate["similarity"],
            )

            if "rerank_score" in candidate:
                span.set_attribute(
                    "retrieval.rerank_score",
                    candidate["rerank_score"],
                )

    # 5. Select top results and expand neighbors around the best result.
    best_rerank_score = (
    candidates[0]["rerank_score"]
    if rerank and candidates
    else None
)
    results = []

    with SessionLocal() as session:
        for candidate in candidates:
            if (
                best_rerank_score is not None
                and candidate["rerank_score"]
                < best_rerank_score - RERANK_SCORE_MARGIN
                ):
                break
            chunk = candidate["_chunk"]

            # Avoid returning chunks already included in another result's
            # expanded context.
            if any(
                existing["source"] == candidate["source"]
                and existing["page_number"] == candidate["page_number"]
                and existing["metadata"]["section_title"]
                == candidate["metadata"]["section_title"]
                and chunk.chunk_index in existing["context_chunk_indices"]
                for existing in results
            ):
                continue

            if any(
                existing["text"] == candidate["text"]
                for existing in results
            ):
                continue

            if not results:
                context_text, context_indices = _build_context(
                    session,
                    chunk,
                    neighbor_window,
                )
            else:
                context_text = chunk.text.strip()
                context_indices = [chunk.chunk_index]

            result = {
                "source": candidate["source"],
                "page_number": candidate["page_number"],
                "chunk_index": candidate["chunk_index"],
                "text": candidate["text"],
                "context_text": context_text,
                "context_chunk_indices": context_indices,
                "metadata": candidate["metadata"],
                "similarity": candidate["similarity"],
            }

            if "rerank_score" in candidate:
                result["rerank_score"] = round(
                    candidate["rerank_score"],
                    4,
                )

            results.append(result)

            with logfire.span("retrieval.result_diagnostic") as span:
                span.set_attribute(
                    "retrieval.query_rank",
                    candidate["_query_rank"],
                )
                span.set_attribute(
                    "retrieval.result_rank",
                    len(results),
                )
                span.set_attribute(
                    "retrieval.similarity",
                    candidate["similarity"],
                )
                span.set_attribute(
                    "retrieval.context_char_count",
                    len(context_text),
                )
                span.set_attribute(
                    "retrieval.context_chunk_count",
                    len(context_indices),
                )

                if "rerank_score" in candidate:
                    span.set_attribute(
                        "retrieval.rerank_score",
                        candidate["rerank_score"],
                    )

            if len(results) >= top_k:
                break

    return results
