from app.guardrails.jailbreaks import (
    is_hr_policy_question,
    is_jailbreak_attempt,
)

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