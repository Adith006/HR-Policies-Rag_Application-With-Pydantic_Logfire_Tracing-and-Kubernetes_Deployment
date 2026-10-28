
import re

JAILBREAK_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above|earlier|system)\s+instructions\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions\b",
    r"\b(reveal|show|print|expose|repeat)\b.{0,60}\b(system prompt|hidden prompt|internal instructions)\b",
    r"\b(bypass|disable|override|remove|ignore)\b.{0,60}\b(guardrails|safety rules|restrictions|instructions)\b",
    r"\bact as\b.{0,40}\b(unrestricted|unfiltered|uncensored|jailbroken)\b",
    r"\b(developer mode|DAN mode|jailbreak mode)\b",
    r"\bpretend\b.{0,50}\b(no rules|no restrictions|without safeguards)\b",
    r"\b(do not follow|stop following)\b.{0,40}\b(system|developer|safety)\s+(instructions|rules)\b",
    r"\b(output|print|reveal|leak)\b.{0,40}\b(api key|secret key|credentials)\b",
]

def is_jailbreak_attempt(text: str) -> bool:
    """Detect common direct or embedded jailbreak phrases."""
    if not isinstance(text, str):
        return False

    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return any(re.search(pattern, normalized) for pattern in JAILBREAK_PATTERNS)


def contains_suspicious_document_instructions(context: str) -> bool:
    """Flag retrieved text containing common instruction-override patterns."""
    return is_jailbreak_attempt(context)
