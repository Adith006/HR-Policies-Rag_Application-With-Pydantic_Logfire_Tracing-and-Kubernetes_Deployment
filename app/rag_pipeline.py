
import hashlib
import json

import logfire

from app.cache.redis_cache import get_cached_answer, cache_answer
from app.guardrails.grounding import validate_grounding
from app.guardrails.jailbreaks import contains_suspicious_document_instructions
from app.guardrails.validation import (
    validate_question,
    QuestionRejectedError,
    HR_POLICY_REFUSAL,
)
from app.llm.client import generate_answer, extract_long_term_memories
from app.memory import ConversationMemory
from app.memory_store import search_memories, store_memory
from app.retrieval.search import semantic_search

conversation_memory = ConversationMemory()


def save_long_term_memories(
    question: str,
    answer: str,
    user_id: str,
    session_id: str,
) -> None:
    """Extract and save useful long-term memories without breaking RAG."""

    with logfire.span("rag.long_term_memory.save") as save_span:
        try:
            with logfire.span("llm.extract_long_term_memories") as span:
                memories = extract_long_term_memories(
                    question=question,
                    answer=answer,
                )
                span.set_attribute(
                    "memory.extracted_count",
                    len(memories) if memories else 0,
                )

            stored_count = 0

            with logfire.span("storage.long_term_memory") as span:
                for memory_text in memories:
                    # Never attach the memory text or user ID to the span.
                    with logfire.span("storage.long_term_memory_item"):
                        store_memory(
                            user_id=user_id,
                            session_id=session_id,
                            memory_text=memory_text,
                        )
                    stored_count += 1

                span.set_attribute("memory.stored_count", stored_count)

            save_span.set_attribute("memory.save_succeeded", True)

        except Exception as exc:
            # This function intentionally preserves the original
            # best-effort behavior: memory failures do not break RAG.
            save_span.set_attribute("memory.save_succeeded", False)
            save_span.set_attribute("error.type", type(exc).__name__)
            logfire.warning(
                "Long-term memory storage failed: {error_type}",
                error_type=type(exc).__name__,
            )


def answer_question(
    question: str,
    session_id: str,
    user_id: str,
) -> dict:
    """Answer a question using RAG, guardrails, caching, STM, and LTM."""

    # Do not include the question, user ID, or session ID as span attributes.
    with logfire.span("rag.validate_question") as span:
        try:
            question = validate_question(question)
            span.set_attribute("question.length", len(question))
            span.set_attribute("guardrail.blocked", False)

        except QuestionRejectedError:
            span.set_attribute("guardrail.blocked", True)
            span.set_attribute("guardrail.reason", "jailbreak_attempt")

            return {
                "question": question,
                "answer": HR_POLICY_REFUSAL,
                "sources": [],
                "cache": "BYPASS",
            }

    # 1. Load short-term conversation history.
    with logfire.span("memory.short_term.retrieve") as span:
        history = conversation_memory.get_history(session_id)
        span.set_attribute(
            "memory.message_count",
            len(history) if history else 0,
        )

    # 2. Retrieve relevant long-term memories.
    with logfire.span("memory.long_term.retrieve") as span:
        long_term_memories = search_memories(
            user_id=user_id,
            query=question,
            top_k=3,
        )
        span.set_attribute(
            "memory.result_count",
            len(long_term_memories) if long_term_memories else 0,
        )

    # 3. Define retrieval settings.
    top_k = 3
    document_type = "HR Policy"
    neighbor_window = 1

    with logfire.span("rag.cache_key.build") as span:
        cache_key = "rag:answer:" + hashlib.sha256(
            json.dumps(
                {
                    "question": question.lower(),
                    "top_k": top_k,
                    "document_type": document_type,
                    "neighbor_window": neighbor_window,
                    "version": 8,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        span.set_attribute("cache.version", 8)

    # 4. Check the shared cache only for first-turn questions.
    cached_data = None

    with logfire.span("cache.answer.lookup") as span:
        if history:
            span.set_attribute(
                "cache.status",
                "skipped_conversation_has_history",
            )
        else:
            cached_data = get_cached_answer(cache_key)
            span.set_attribute(
                "cache.status",
                "hit" if cached_data else "miss",
            )

    if cached_data:
        with logfire.span("cache.answer.validate") as span:
            answer = cached_data["answer"]
            sources = cached_data["sources"]
            context = cached_data["context"]

            if not answer.strip():
                raise ValueError("Cached answer is empty.")

            if not sources:
                raise ValueError("Cached answer has no source citations.")

            with logfire.span("guardrails.cached_answer_grounding") as grounding_span:
                grounded = validate_grounding(answer, context)
                grounding_span.set_attribute("grounding.valid", grounded)

            if not grounded:
                raise ValueError(
                    "Cached answer failed grounding validation."
                )

            span.set_attribute("citation.count", len(sources))

        with logfire.span("memory.short_term.save_cached_turn"):
            conversation_memory.add_turn(
                session_id, question, answer
            )

        save_long_term_memories(
            question, answer, user_id, session_id
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "cache": "HIT",
        }

    # 5. Retrieve relevant document chunks.
    with logfire.span("retrieval.semantic_search") as span:
        results = semantic_search(
            question=question,
            top_k=top_k,
            document_type=document_type,
            neighbor_window=neighbor_window,
        )

        span.set_attribute(
            "retrieval.result_count",
            len(results) if results else 0,
        )
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.neighbor_window", neighbor_window)

    if not results:
        answer = (
            "I couldn't find relevant information "
            "in the available documents."
        )

        with logfire.span("rag.fallback.no_retrieval_results"):
            with logfire.span("memory.short_term.save_fallback_turn"):
                conversation_memory.add_turn(
                    session_id, question, answer
                )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "cache": "MISS",
        }

    # 6. Build the context from retrieved chunks.
    with logfire.span("retrieval.context.build") as span:
        context = "\n\n".join(
            result["context_text"]
            for result in results
            if result["context_text"].strip()
        )

        # Record only its length, never the actual document text.
        span.set_attribute("context.length", len(context))
        span.set_attribute("context.has_content", bool(context))

    if not context:
        answer = (
            "The retrieved documents contained no usable context."
        )

        with logfire.span("rag.fallback.empty_context"):
            with logfire.span("memory.short_term.save_fallback_turn"):
                conversation_memory.add_turn(
                    session_id, question, answer
                )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "cache": "MISS",
        }

    # 7. Check for suspicious instructions in retrieved documents.
    with logfire.span("guardrails.document_instruction_check") as span:
        suspicious = contains_suspicious_document_instructions(context)
        span.set_attribute("guardrail.blocked", suspicious)

        if suspicious:
            raise ValueError(
                "Retrieved context contains suspicious instructions. "
                "Answer generation blocked."
            )

    # 8. Generate an answer using short-term and long-term memory.
    with logfire.span("llm.answer_generation") as span:
        answer = generate_answer(
            question=question,
            context=context,
            history=history,
            long_term_memories=long_term_memories,
        )

        span.set_attribute(
            "llm.answer.length",
            len(answer) if isinstance(answer, str) else 0,
        )

    # 9. Validate the answer against the retrieved context.
    with logfire.span("guardrails.grounding_validation") as span:
        grounded = validate_grounding(answer, context)
        span.set_attribute("grounding.valid", grounded)

    if not grounded:
        answer = (
            "I couldn't verify that the generated answer is sufficiently "
            "supported by the available document context."
        )

        with logfire.span("rag.fallback.grounding_failure"):
            with logfire.span("memory.short_term.save_fallback_turn"):
                conversation_memory.add_turn(
                    session_id, question, answer
                )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "cache": "MISS",
        }

    # 10. Build source citations.
    with logfire.span("retrieval.citations.build") as span:
        sources = [
            {
                "source": result["source"],
                "page_number": result["page_number"],
                "section_title": result["metadata"]["section_title"],
                "similarity": result["similarity"],
            }
            for result in results
        ]

        span.set_attribute("citation.count", len(sources))

    # 11. Cache standalone answers.
    if not history:
        with logfire.span("cache.answer.save"):
            cache_answer(cache_key, answer, sources, context)

    # 12. Save the conversation to short-term memory.
    with logfire.span("memory.short_term.save_turn"):
        conversation_memory.add_turn(
            session_id, question, answer
        )

    # 13. Extract and save long-term memories.
    save_long_term_memories(
        question=question,
        answer=answer,
        user_id=user_id,
        session_id=session_id,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "cache": "MISS",
    }
