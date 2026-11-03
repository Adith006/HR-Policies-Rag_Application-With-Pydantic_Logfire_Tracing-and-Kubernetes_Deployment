from app.guardrails.jailbreaks import (
    is_hr_policy_question,
    is_jailbreak_attempt,
)
# ============================================================
# PROFESSIONAL GREETING HANDLER
# ============================================================
import re


GREETING_RESPONSES = [
    (
        r"(?:hi|hello|hey|greetings)(?: there)?",
        "Hello! I'm the HR Policy Assistant. "
        "I can help you find information from the organization's "
        "policy manual. What would you like to know?",
    ),
    (
        r"good morning",
        "Good morning! Welcome to the HR Policy Assistant. "
        "How may I assist you with the organization's policy manual?",
    ),
    (
        r"good afternoon",
        "Good afternoon! I'm here to help with questions "
        "about the organization's HR policies. What would you like to know?",
    ),
    (
        r"good evening",
        "Good evening! Welcome. "
        "How can I help you with the organization's HR policies today?",
    ),
    (
        r"how are you(?: doing)?",
        "I'm doing well, thank you. "
        "I'm ready to help you with questions about the HR policy manual. "
        "How can I assist you?",
    ),
    (
        r"(?:thanks|thank you)(?: very much| so much| a lot)?",
        "You're welcome! I'm happy to help with your HR policy questions.",
    ),
    (
        r"(?:bye|goodbye|see you|see you later|good night)",
        "Thank you for using the HR Policy Assistant. "
        "Have a pleasant day!",
    ),
]


def get_greeting_response(text: str) -> str | None:
    """
    Return a professional response for standalone greetings.

    Only exact greeting messages are accepted.
    Mixed greetings and policy questions continue through
    the normal HR policy validation pipeline.
    """

    if not isinstance(text, str) or not text.strip():
        return None

    normalized = re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )

    normalized = re.sub(
        r"[.!?,]+$",
        "",
        normalized,
    ).strip()

    for pattern, response in GREETING_RESPONSES:

        if re.fullmatch(pattern, normalized):
            return response

    return None

HR_POLICY_REFUSAL = (
    "I can only help with questions about HR policies covered by "
    "the available documents. Please ask an HR-policy-related question."
)


class QuestionRejectedError(ValueError):
    """Raised when a security or topic-scope guardrail rejects input."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def validate_question(question: str) -> str:
    """Validate input, detect jailbreaks, and enforce HR-policy scope."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    if len(question) > 2000:
        raise ValueError("Question exceeds the 2000-character limit.")

    if is_jailbreak_attempt(question):
        raise QuestionRejectedError(
            HR_POLICY_REFUSAL,
            reason="jailbreak_attempt",
        )

    if not is_hr_policy_question(question):
        raise QuestionRejectedError(
            HR_POLICY_REFUSAL,
            reason="out_of_scope",
        )

    return question