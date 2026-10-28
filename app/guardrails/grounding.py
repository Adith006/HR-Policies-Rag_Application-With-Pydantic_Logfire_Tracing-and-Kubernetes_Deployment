
import re


def _words(text: str) -> list[str]:
    """Extract normalized words while preserving repeated words."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def validate_grounding(answer: str, context: str) -> bool:
    """Apply a basic lexical-support check to each answer sentence."""
    if not isinstance(answer, str) or not answer.strip():
        return False

    if not isinstance(context, str) or not context.strip():
        return False

    context_words = set(_words(context))
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    for sentence in sentences:
        sentence_words = _words(sentence)

        # Ignore short sentences that contain little factual information.
        if len(sentence_words) < 4:
            continue

        # Calculate the proportion of answer words found in the context.
        overlap = sum(
            1 for word in sentence_words if word in context_words
        )
        overlap_ratio = overlap / len(sentence_words)
        # print(
        #     f"[GROUNDING DEBUG] words={len(sentence_words)}, "
        #     f"matched={overlap}, "
        #     f"overlap_ratio={overlap_ratio:.3f}"
        # )
        # print(f"[GROUNDING DEBUG] Rejected candidate: {sentence!r}"
        #       if overlap_ratio < 0.5
        #       else "[GROUNDING DEBUG] Sentence passed")

        if overlap_ratio < 0.5:
            # print(
            #     f"[GROUNDING DEBUG] Rejected sentence "
            #     f"(overlap={overlap_ratio:.2f}, "
            #     f"threshold=0.50)"
            #     )
            # print(f"[GROUNDING DEBUG] Sentence: {sentence}")
            return False



    return True
