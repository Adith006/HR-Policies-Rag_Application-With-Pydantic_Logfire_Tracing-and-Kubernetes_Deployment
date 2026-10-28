
from app.guardrails.jailbreaks import is_jailbreak_attempt


def validate_question(question: str) -> str:
    """Validate user input and reject common jailbreak attempts."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    if len(question) > 2000:
        raise ValueError("Question exceeds the 2000-character limit.")

    if is_jailbreak_attempt(question):
        raise ValueError("Question rejected by the security guardrail.")

    return question
