
import re
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model only when chunking needs it."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model



def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    lines: list[dict] | None = None,
    initial_section: str = "Unknown",
) -> list[dict]:
    """Create chunks with section titles extracted from PDF lines."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size")

    if not text.strip():
        return []

    units = []
    current_section = initial_section

    if lines:
        merged_lines = []

        known_heading_patterns = [
            "reporting and resolution of violations",
            "equal employment, non-discrimination",
            "harassment, bullying, violence in the workplace policy",
        ]

        # Detect headings without treating bullet points as headings.
        for line in lines:
            value = line["text"].strip()

            if not value:
                continue

            value_lower = value.lower()
            is_bullet = value.startswith(("•", "–", "-", "▪"))

            is_heading = (
                (
                    line.get("font_size", 10) >= 11.5
                    and not is_bullet
                )
                or (
                    line.get("is_bold", False)
                    and not is_bullet
                    and len(value) < 100
                    and not value.endswith((",", ";", ":", "–", "-"))
                )
                or any(
                    pattern in value_lower
                    for pattern in known_heading_patterns
                )
            )

            # Merge short wrapped lines belonging to a heading.
            if (
                merged_lines
                and merged_lines[-1]["is_heading"]
                and not is_heading
                and len(merged_lines[-1]["text"]) < 100
                and not re.search(
                    r"[.!?;:]$", merged_lines[-1]["text"]
                )
                and len(value) < 50

                # NEW: avoid merging smaller-font body text into a heading.
                and line.get("font_size", 10)
                >= merged_lines[-1]["font_size"] - 1.0

                 # NEW: avoid merging complete sentences into headings.
                and not re.search(r"[.!?;:]$", value)
            ):
                merged_lines[-1]["text"] += " " + value
            else:
                merged_lines.append({
                    "text": value,
                    "font_size": line.get("font_size", 10),
                    "is_bold": line.get("is_bold", False),
                    "is_heading": is_heading,
                })

        # Recognize the wrapped Equal Employment heading.
        for line in merged_lines:
            if "equal employment, non-discrimination" in line["text"].lower():
                line["is_heading"] = True

        # Build content units, keeping headings out of body text.
        i = 0

        while i < len(merged_lines):
            line = merged_lines[i]
            value = line["text"].strip()

            if (
                line["is_heading"]
                and value.endswith("Accommo-")
                and i + 1 < len(merged_lines)
                and merged_lines[i + 1]["text"].strip().lower() == "dation"
            ):
                value = value[:-1] + merged_lines[i + 1]["text"].strip()
                i += 1

            if line["is_heading"]:
                current_section = value
                i += 1
                continue

            sentences = re.split(r"(?<=[.!?])\s+", value)

            for sentence in sentences:
                if sentence.strip():
                    units.append({
                        "text": sentence.strip(),
                        "section_title": current_section,
                    })

            i += 1

    else:
        # Fallback when PDF line metadata is unavailable.
        paragraphs = re.split(r"\n\s*\n", text.strip())

        for paragraph in paragraphs:
            paragraph = re.sub(r"\s+", " ", paragraph).strip()

            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if sentence.strip():
                    units.append({
                        "text": sentence.strip(),
                        "section_title": initial_section,
                    })

    # Split oversized units at word boundaries.
    split_units = []

    for unit in units:
        part = ""

        for word in unit["text"].split():
            candidate = f"{part} {word}".strip()

            if len(candidate) > chunk_size and part:
                split_units.append({**unit, "text": part})
                part = word
            else:
                part = candidate

        if part:
            split_units.append({**unit, "text": part})

    if not split_units:
        return []

    # Generate embeddings for semantic boundary detection.
    embeddings =  _get_model().encode(
        [unit["text"] for unit in split_units],
        normalize_embeddings=True,
    )

    # Build chunks without crossing section boundaries.
    chunks = []
    current_text = ""
    current_section = split_units[0]["section_title"]

    for i, unit in enumerate(split_units):
        value = unit["text"]
        section = unit["section_title"]

        if current_text and section != current_section:
            chunks.append({
                "text": current_text.strip(),
                "section_title": current_section,
            })
            current_text = ""

        candidate = f"{current_text} {value}".strip()

        if current_text and len(candidate) > chunk_size:
            chunks.append({
                "text": current_text.strip(),
                "section_title": current_section,
            })
            current_text = ""

        if current_text and i > 0:
            similarity = float(
                np.dot(embeddings[i - 1], embeddings[i])
            )

            if (
                similarity < 0.45
                and len(current_text) >= chunk_size // 2
            ):
                chunks.append({
                    "text": current_text.strip(),
                    "section_title": current_section,
                })
                current_text = ""

        if not current_text:
            current_section = section
            current_text = value
        else:
            current_text = f"{current_text} {value}"

    if current_text:
        chunks.append({
            "text": current_text.strip(),
            "section_title": current_section,
        })

    # Add word-based overlap only within the same section.
    output = []

    for i, chunk in enumerate(chunks):
        value = chunk["text"]

        if (
            i > 0
            and chunk_overlap > 0
            and chunks[i - 1]["section_title"] == chunk["section_title"]
        ):
            overlap = ""

            for word in reversed(chunks[i - 1]["text"].split()):
                candidate = f"{word} {overlap}".strip()

                if len(candidate) > chunk_overlap:
                    break

                overlap = candidate

            candidate = f"{overlap} {value}".strip()

            if len(candidate) <= chunk_size:
                value = candidate

        output.append({
            "text": value,
            "section_title": chunk["section_title"],
        })

    return output
