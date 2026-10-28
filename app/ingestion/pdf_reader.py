
import fitz
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean PDF text while preserving paragraph and line boundaries."""
    text = text.replace("\x00", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_path: str) -> list[dict]:
    """Extract page text and retain font-aware line information."""
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("dict")["blocks"]
            lines = []

            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span["text"] for span in spans).strip()

                    if text:
                        lines.append({
                            "text": text,
                            "font_size": max(span["size"] for span in spans),
                            "is_bold": any(
                                "bold" in span["font"].lower()
                                for span in spans
                            ),
                        })

            text = clean_text(page.get_text("text"))

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "lines": lines,
                })

    return pages
