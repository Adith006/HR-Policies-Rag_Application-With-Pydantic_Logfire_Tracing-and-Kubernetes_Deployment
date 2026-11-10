
"""Safe, user-facing messages for application request failures."""

import httpx


def user_facing_error_message(
    exc: Exception,
    operation: str = "answer",
) -> str:
    """Return a safe message without exposing provider details."""

    is_evaluation = operation == "evaluation"

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code

        if status_code in {402, 408, 425, 429} or status_code >= 500:
            if is_evaluation:
                return (
                    "The evaluation service is temporarily unavailable "
                    "or has reached a usage limit. Please try again later."
                )

            return (
                "The AI response service is temporarily unavailable or "
                "has reached a usage limit. No answer was generated. "
                "Please try again in a few minutes."
            )

        if status_code in {401, 403}:
            return (
                "Your session could not be verified. "
                "Please sign out and sign in again."
            )

        if 400 <= status_code < 500:
            return (
                "The request could not be completed. "
                "Please check your input and try again."
            )

    if isinstance(exc, httpx.TransportError):
        if is_evaluation:
            return (
                "The evaluation service could not be reached. "
                "Please try again shortly."
            )

        return (
            "The AI service could not be reached. "
            "Please try again shortly."
        )

    if is_evaluation:
        return (
            "The evaluation could not be completed. "
            "Please try again later."
        )

    return (
        "We couldn't generate an answer right now. "
        "Please try again later."
    )
