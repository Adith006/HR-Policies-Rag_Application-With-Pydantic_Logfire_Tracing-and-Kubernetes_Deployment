import hashlib
import json

import logfire
from app.cache.redis_cache import get_cached_answer, cache_answer
from app.guardrails.grounding import validate_grounding
from app.guardrails.jailbreaks import contains_suspicious_document_instructions
from app.guardrails.validation import (
    get_greeting_response,
    validate_question,
    QuestionRejectedError,
    HR_POLICY_REFUSAL,
)
from app.llm.client import generate_answer,summarize_hr_conversation
from app.memory import ConversationMemory
from app.memory_store import search_memories, store_memory
from app.retrieval.search import semantic_search
from app.rag_trace import trace_stage
import re


conversation_memory = ConversationMemory()

def classify_question_intent(question: str) -> str:
    """
    Classify questions into historical, combined or current policy.
    Current policy is the default when intent is ambiguous.
    """

    historical_patterns = [
        r"\bpreviously\b",
        r"\bprevious discussion\b",
        r"\bprevious conversation\b",
        r"\bearlier discussion\b",
        r"\bdiscussed earlier\b",
        r"\bdiscussed previously\b",
        r"\bwhat did we discuss\b",
        r"\bwhat did we talk about\b",
        r"\bwhat did we establish\b",
        r"\bremind me\b",
        r"\brecall\b",
        r"\bfrom our last conversation\b",
        r"\bfrom our previous discussion\b",
    ]

    combined_patterns = [
    r"\bhas .{0,50} changed\b",
    r"\bchanged since\b",
    r"\bcurrent(?:ly)?\s+(?:policy|status|state|says|states|provide|provides|mention|mentions)\b",
    r"\bwhat does .{0,80} currently (?:state|say|specify|mention)\b",
    r"\bis .{0,40} still\b",
    r"\bcompare .{0,50} current\b",
    r"\bcompared with .{0,50} current\b",
    r"\bwhat is the current\b",
]



    is_historical = any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in historical_patterns
    )

    is_combined = any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in combined_patterns
    )

    if is_historical and is_combined:
        return "combined"

    if is_historical:
        return "historical"

    return "current"

def get_retrieval_queries(question: str, intent: str) -> tuple[str, str]:
    """
    Return separate historical and document retrieval queries.
    """
    if intent != "combined":
        return question, question

    match = re.search(
        r"\b(?:and|but)\s+(?=(?:what|how|does|do|is|are|has|have)\b"
        r".{0,180}\b(?:current(?:ly)?|still|changed|compare|compared)\b)",
        question,
        re.IGNORECASE,
    )

    if not match:
        return question, question

    historical_query = question[:match.start()].strip(" ,;")
    document_query = question[match.end():].strip(" ,;")

    return (
        historical_query or question,
        document_query or question,
    )


def validate_combined_grounding(
    answer: str,
    document_context: str,
    long_term_memories: list[dict],
) -> bool:
    """
    Validate historical and current-policy answer sections separately.

    Expected headings are supplied by the LLM prompt:
    - Previous discussion (historical):
    - Current HR manual:
    """
    if not isinstance(answer, str) or not answer.strip():
        return False

    historical_match = re.search(
        r"(?is)previous discussion\s*\(historical\)\s*:\s*(.*?)(?="
        r"\n\s*current HR manual\s*:|\Z)",
        answer,
    )
    current_match = re.search(
        r"(?is)current HR manual\s*:\s*(.*)\Z",
        answer,
    )

    if not historical_match or not current_match:
        return False

    historical_section = historical_match.group(1).strip()
    current_section = current_match.group(1).strip()

    if not current_section or not document_context.strip():
        return False

    # If no historical memory was retrieved, only accept an explicit
    # statement that no relevant prior discussion was found.
    if not long_term_memories:
        missing_history_is_explicit = bool(
            re.search(
                r"(?i)(couldn't find|could not find|no relevant "
                r"(record|previous discussion)|unable to find)",
                historical_section,
            )
        )
        historical_grounded = missing_history_is_explicit
    else:
        memory_context = "\n\n".join(
            item.get("memory", "")
            for item in long_term_memories
            if isinstance(item, dict)
            and isinstance(item.get("memory"), str)
            and item.get("memory", "").strip()
        )
        historical_grounded = bool(memory_context) and validate_grounding(
            historical_section,
            memory_context,
        )

    current_grounded = validate_grounding(
        current_section,
        document_context,
    )

    return historical_grounded and current_grounded


def save_hr_conversation_summary(
    question: str,
    answer: str,
    user_id: str,
    session_id: str,
    sources: list[dict],
) -> None:
    """Save a summary of a successfully grounded HR conversation."""

    try:
        summary_data = summarize_hr_conversation(
            question=question,
            answer=answer,
        )

        if not summary_data:
            return

        topic = summary_data["topic"]
        summary = summary_data["summary"]

        memory_text = (
            f"Topic: {topic}\n"
            f"Summary: {summary}"
        )

        source_documents = [
            {
                "source": item.get("source"),
                "page_number": item.get("page_number"),
                "section_title": item.get("section_title"),
            }
            for item in sources
        ]

        store_memory(
            user_id=user_id,
            session_id=session_id,
            memory_text=memory_text,
            memory_type="hr_conversation_summary",
            metadata={
                "topic": topic,
                "source": "hr_policy_conversation",
                "source_documents": source_documents,
            },
        )

    except Exception as exc:
        logfire.warning(
            "HR conversation memory save failed: {error_type}",
            error_type=type(exc).__name__,
        )

def build_retrieved_contexts(results: list[dict]) -> list[dict]:
    """
    Convert semantic retrieval results into UI-friendly context objects.

    Each object contains the actual retrieved text and its metadata.
    """

    retrieved_contexts = []

    for result in results or []:

        context_text = (
            result.get("context_text")
            or result.get("text")
            or ""
        )

        if not context_text.strip():
            continue

        metadata = result.get("metadata") or {}

        retrieved_contexts.append(
            {
                "text": context_text,
                "source": result.get("source"),
                "page_number": result.get("page_number"),
                "section_title": metadata.get("section_title"),
                "similarity": result.get("similarity"),
                "chunk_index": result.get("chunk_index"),
                "context_chunk_indices": result.get(
                    "context_chunk_indices",
                    [],
                ),
            }
        )

    return retrieved_contexts


def answer_question(
    question: str,
    session_id: str,
    user_id: str,
) -> dict:
    """
    Answer a question using RAG, guardrails, caching,
    short-term memory and long-term memory.

    Returns answer, sources, cache status and retrieved contexts.
    """
    # ============================================================
    # 0. PROFESSIONAL GREETING HANDLING
    # ============================================================

    with trace_stage("Professional greeting handling") as trace:

        with logfire.span("rag.greeting_detection") as span:

            greeting_answer = get_greeting_response(question)

            span.set_attribute(
                "greeting.detected",
                greeting_answer is not None,
            )

            if greeting_answer:

                return {
                    "question": question,
                    "answer": greeting_answer,
                    "sources": [],
                    "retrieved_contexts": [],
                    "cache": "BYPASS",
                }

    # ============================================================
    # 1. QUESTION VALIDATION
    # ============================================================

    with trace_stage("Question validation") as trace:

        with logfire.span("rag.validate_question") as span:

            try:
                question = validate_question(question)

                span.set_attribute(
                    "question.length",
                    len(question),
                )

                span.set_attribute(
                    "guardrail.blocked",
                    False,
                )

            except QuestionRejectedError as exc:

                trace["status"] = "blocked"
                trace["reason"] = exc.reason

                span.set_attribute(
                    "guardrail.blocked",
                    True,
                )

                span.set_attribute(
                    "guardrail.reason",
                    exc.reason,
                )

                return {
                    "question": question,
                    "answer": HR_POLICY_REFUSAL,
                    "sources": [],
                    "retrieved_contexts": [],
                    "cache": "BYPASS",
                }

    # ============================================================
    # 2. SHORT-TERM MEMORY RETRIEVAL
    # ============================================================

    with trace_stage("Short-term memory retrieval") as trace:

        with logfire.span("memory.short_term.retrieve") as span:

            # history = conversation_memory.get_history(session_id)
            history = conversation_memory.get_history(user_id, session_id)

            span.set_attribute(
                "memory.message_count",
                len(history) if history else 0,
            )

    # ============================================================
    # 3. QUESTION INTENT AND RETRIEVAL QUERY SPLITTING
    # ============================================================

    intent = classify_question_intent(question)
    historical_query, document_query = get_retrieval_queries(
        question,
        intent,
    )

    with logfire.span("rag.question_intent") as span:
        span.set_attribute("question.intent", intent)

    # ============================================================
    # 4. LONG-TERM MEMORY RETRIEVAL
    # ============================================================

    with trace_stage("Long-term memory retrieval") as trace:

        with logfire.span("memory.long_term.retrieve") as span:

            long_term_memories = []

            if intent in ("historical", "combined"):
                long_term_memories = search_memories(
                    user_id=user_id,
                    query=historical_query,
                    top_k=3,
                )

            span.set_attribute(
                "memory.result_count",
                len(long_term_memories),
            )



    if intent == "historical":

        if not long_term_memories:
            answer = (
                "I couldn't find a relevant record of that topic "
                "in our previous conversations."
            )

            conversation_memory.add_turn(
                user_id,
                session_id,
                question,
                answer,
            )

            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "retrieved_contexts": [],
                "cache": "BYPASS",
            }

        memory_context = "\n\n".join(
            memory["memory"]
            for memory in long_term_memories
            if isinstance(memory.get("memory"), str)
            and memory["memory"].strip()
        )

        if not memory_context.strip():
            raise ValueError("Retrieved historical memories are empty.")

        with trace_stage("Historical answer generation"):

            answer = generate_answer(
                question=question,
                context="",
                history=history,
                long_term_memories=long_term_memories,
                intent="historical",
            )

        with trace_stage("Historical grounding validation"):

            grounded = validate_grounding(
                answer,
                memory_context,
            )

        if not grounded:
            answer = (
                "I couldn't verify that answer against the "
                "available historical conversation summaries."
            )

            conversation_memory.add_turn(
                user_id,
                session_id,
                question,
                answer,
            )

            return {
                "question": question,
                "answer": answer,
                "sources": [],
                "retrieved_contexts": [],
                "cache": "BYPASS",
            }

        conversation_memory.add_turn(
            user_id,
            session_id,
            question,
            answer,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "retrieved_contexts": [
                {
                    "text": memory["memory"],
                    "source": "conversation_memory",
                    "session_id": memory.get("session_id"),
                    "memory_id": memory.get("id"),
                    "similarity": memory.get("similarity"),
                }
                for memory in long_term_memories
            ],
            "cache": "BYPASS",
        }

    # ============================================================
    # 4. RETRIEVAL CONFIGURATION AND CACHE KEY
    # ============================================================

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
                    "version": 10,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        span.set_attribute(
            "cache.version",
            10,
        )

    # ============================================================
    # 5. ANSWER CACHE LOOKUP
    # ============================================================

    cached_data = None

    with trace_stage("Answer cache lookup") as trace:

        with logfire.span("cache.answer.lookup") as span:

            if history or intent != "current":

                span.set_attribute(
                    "cache.status",
                    "skipped_non_current_or_conversation_has_history",
                )

            else:

                cached_data = get_cached_answer(cache_key)

                span.set_attribute(
                    "cache.status",
                    "hit" if cached_data else "miss",
                )

    # ------------------------------------------------------------
    # 5A. RETURN CACHED ANSWER
    # ------------------------------------------------------------

    if cached_data:

        with logfire.span("cache.answer.validate") as span:

            answer = cached_data["answer"]
            sources = cached_data["sources"]
            context = cached_data["context"]

            # Support cached context collections when available.
            retrieved_contexts = (
                cached_data.get("retrieved_contexts") or []
            )

            # Existing cache entries contain the combined context.
            # Display that context as one item when individual
            # retrieved chunks were not stored in Redis.
            if not retrieved_contexts and context.strip():

                retrieved_contexts = [
                    {
                        "text": context,
                        "section_title": "Combined retrieved context",
                    }
                ]

            if not answer.strip():
                raise ValueError("Cached answer is empty.")

            if not sources:
                raise ValueError(
                    "Cached answer has no source citations."
                )

            with logfire.span(
                "guardrails.cached_answer_grounding"
            ) as grounding_span:

                grounded = validate_grounding(
                    answer,
                    context,
                )

                grounding_span.set_attribute(
                    "grounding.valid",
                    grounded,
                )

            if not grounded:
                raise ValueError(
                    "Cached answer failed grounding validation."
                )

            span.set_attribute(
                "citation.count",
                len(sources),
            )

        # with logfire.span("memory.short_term.save_cached_turn"):

        #     conversation_memory.add_turn(
        #         session_id,
        #         question,
        #         answer,
        #     )

        # save_long_term_memories(
        #     question,
        #     answer,
        #     user_id,
        #     session_id,
        # )
        conversation_memory.add_turn(
            user_id,
            session_id,
            question,
            answer,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
            "cache": "HIT",
        }

    # ============================================================
    # 6. SEMANTIC RETRIEVAL AND RERANKING
    # ============================================================

    with trace_stage("Semantic retrieval and reranking") as trace:

        with logfire.span("retrieval.semantic_search") as span:

            results = semantic_search(
                question=document_query,
                top_k=top_k,
                document_type=document_type,
                neighbor_window=neighbor_window,
            )

            span.set_attribute(
                "retrieval.result_count",
                len(results) if results else 0,
            )

            span.set_attribute(
                "retrieval.top_k",
                top_k,
            )

            span.set_attribute(
                "retrieval.neighbor_window",
                neighbor_window,
            )

    # IMPORTANT:
    # Build retrieved contexts immediately after retrieval.
    # Do this before checking whether results are empty.

    retrieved_contexts = build_retrieved_contexts(
        results or []
    )

    # ============================================================
    # 7. HANDLE EMPTY RETRIEVAL RESULTS
    # ============================================================

    if not results:

        answer = (
            "I couldn't find relevant information "
            "in the available documents."
        )

        with logfire.span("rag.fallback.no_retrieval_results"):

            with logfire.span(
                "memory.short_term.save_fallback_turn"
            ):

                conversation_memory.add_turn(
                    user_id,
                    session_id,
                    question,
                    answer,
                )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "retrieved_contexts": retrieved_contexts,
            "cache": "MISS",
        }

    # ============================================================
    # 8. BUILD LLM CONTEXT
    # ============================================================

    with trace_stage("Context assembly") as trace:

        with logfire.span("retrieval.context.build") as span:

            context = "\n\n".join(
                item["text"]
                for item in retrieved_contexts
                if item["text"].strip()
            )

            # Never record actual document text in tracing.
            span.set_attribute(
                "context.length",
                len(context),
            )

            span.set_attribute(
                "context.has_content",
                bool(context),
            )

            span.set_attribute(
                "context.retrieved_count",
                len(retrieved_contexts),
            )

    # ============================================================
    # 9. HANDLE EMPTY CONTEXT
    # ============================================================

    if not context.strip():

        answer = (
            "The retrieved documents contained no usable context."
        )

        with logfire.span("rag.fallback.empty_context"):

            with logfire.span(
                "memory.short_term.save_fallback_turn"
            ):

                conversation_memory.add_turn(
                    user_id,
                    session_id,
                    question,
                    answer,
                )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "retrieved_contexts": retrieved_contexts,
            "cache": "MISS",
        }

    # ============================================================
    # 10. RETRIEVED DOCUMENT SAFETY CHECK
    # ============================================================

    with trace_stage("Retrieved document safety check") as trace:

        with logfire.span(
            "guardrails.document_instruction_check"
        ) as span:

            suspicious = (
                contains_suspicious_document_instructions(context)
            )

            span.set_attribute(
                "guardrail.blocked",
                suspicious,
            )

            if suspicious:

                raise ValueError(
                    "Retrieved context contains suspicious "
                    "instructions. Answer generation blocked."
                )

    # ============================================================
    # 11. LLM ANSWER GENERATION
    # ============================================================

    with trace_stage("LLM answer generation") as trace:

        with logfire.span("llm.answer_generation") as span:
            # print("\n========== LLM INPUT DEBUG ==========")
            # print("QUESTION:", question)
            # print("\n--- RETRIEVED CONTEXT ---")
            # print(context)
            # print("\n--- CONVERSATION HISTORY ---")
            # print(history)
            # print("\n--- LONG-TERM MEMORIES ---")
            # print(long_term_memories)
            # print("=====================================\n")
            # # TEMPORARY TEST: Exclude long-term memories
            # long_term_memories = []

            # print("LONG-TERM MEMORIES AFTER OVERRIDE:", long_term_memories)

            answer = generate_answer(
                question=question,
                context=context,
                history=history,
                long_term_memories=long_term_memories,
                intent=intent,
            )
            # print("\n========== GROUNDING DEBUG ==========")
            # print("QUESTION:", question)
            # print("GENERATED ANSWER:", answer)
            # print("====================================\n")

            span.set_attribute(
                "llm.answer.length",
                len(answer) if isinstance(answer, str) else 0,
            )

    # ============================================================
    # 12. GROUNDING VALIDATION
    # ============================================================

    with trace_stage("Grounding validation") as trace:

        with logfire.span(
            "guardrails.grounding_validation"
        ) as span:
            if intent == "combined":
                grounded = validate_combined_grounding(
                    answer=answer,
                    document_context=context,
                    long_term_memories=long_term_memories,
                )
            else:
                grounded = validate_grounding(answer, context)



            span.set_attribute(
                "grounding.valid",
                grounded,
            )

    # ------------------------------------------------------------
    # 12A. HANDLE GROUNDING FAILURE
    # ------------------------------------------------------------

    if not grounded:

        answer = (
            "I couldn't verify that the generated answer is "
            "sufficiently supported by the available document context."
        )

        with logfire.span("rag.fallback.grounding_failure"):

            with logfire.span(
                "memory.short_term.save_fallback_turn"
            ):

                conversation_memory.add_turn(
                    user_id,
                    session_id,
                    question,
                    answer,
                )

        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "retrieved_contexts": retrieved_contexts,
            "cache": "MISS",
        }

    # ============================================================
    # 13. BUILD SOURCE CITATIONS
    # ============================================================

    with trace_stage("Source citation generation") as trace:

        with logfire.span("retrieval.citations.build") as span:

            sources = [
                {
                    "source": result.get("source"),
                    "page_number": result.get("page_number"),
                    "section_title": (
                        (result.get("metadata") or {}).get(
                            "section_title"
                        )
                    ),
                    "similarity": result.get("similarity"),
                }
                for result in results
            ]

            span.set_attribute(
                "citation.count",
                len(sources),
            )
    with logfire.span("memory.long_term.save_hr_summary"):
        save_hr_conversation_summary(
            question=question,
            answer=answer,
            user_id=user_id,
            session_id=session_id,
            sources=sources,
            )

    # ============================================================
    # 14. CACHE ANSWER
    # ============================================================

    if not history and intent == "current":

        with logfire.span("cache.answer.save"):

            # Keep the existing four-argument cache interface.
            # The combined context is stored and can be displayed
            # when this answer is served from cache.
            cache_answer(
                cache_key,
                answer,
                sources,
                context,
                retrieved_contexts=retrieved_contexts,
            )

    # ============================================================
    # 15. SAVE SHORT-TERM MEMORY
    # ============================================================

    with logfire.span("memory.short_term.save_turn"):

        conversation_memory.add_turn(
            user_id,
            session_id,
            question,
            answer,
        )

    # ============================================================
    # 16. SAVE LONG-TERM MEMORY
    # ============================================================

    # save_long_term_memories(
    #     question=question,
    #     answer=answer,
    #     user_id=user_id,
    #     session_id=session_id,
    # )

    # ============================================================
    # 17. RETURN COMPLETE RAG RESPONSE
    # ============================================================

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
        "cache": "MISS",
    }
