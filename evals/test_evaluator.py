import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

# Load .env from the project root, regardless of the current file path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
PORTKEY_CONFIG = "pc-enterp-1a6936"

if not PORTKEY_API_KEY or not GROQ_MODEL:
    raise RuntimeError(
        "PORTKEY_API_KEY or GROQ_MODEL is missing from the project .env file."
    )


def main() -> None:
    """Verify that the Ragas evaluator can call an LLM through Portkey."""

    llm = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=createHeaders(
            api_key=PORTKEY_API_KEY,
            config=PORTKEY_CONFIG,
        ),
        temperature=0,
        max_tokens=512,
    )

    response = llm.invoke(
        "Reply with exactly: evaluator connected"
    )

    print("Evaluator connection successful.")
    print("Model response:", response.content)
    print("Evaluator connection successful.")
    print("Response content:", repr(response.content))
    print("Response metadata:", response.response_metadata)
    print("Additional data:", response.additional_kwargs)


if __name__ == "__main__":
    main()