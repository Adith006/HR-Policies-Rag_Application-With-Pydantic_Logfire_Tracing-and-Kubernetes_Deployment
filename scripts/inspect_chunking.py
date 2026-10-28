import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pipeline import process_pdf

chunks = process_pdf("sample_documents/manual.pdf")

print("Total chunks:", len(chunks))

for chunk in chunks[:5]:
    print(
        f"Chunk {chunk['chunk_index']} | "
        f"Page {chunk['page_number']} | "
        f"Length: {len(chunk['text'])}"
    )
    print("Start:", repr(chunk["text"][:100]))
    print("End:", repr(chunk["text"][-100:]))
    print("-" * 40)
lengths = [len(chunk["text"]) for chunk in chunks]

print("Maximum chunk length:", max(lengths))
print("Chunks over 800 characters:", sum(n > 800 for n in lengths))
print("Chunks under 200 characters:", sum(n < 200 for n in lengths))
print("Unknown section titles:", sum(
    chunk["metadata"]["section_title"] == "Unknown"
    for chunk in chunks
))
unknown_chunks = [
    chunk for chunk in chunks
    if chunk["metadata"]["section_title"] == "Unknown"
]

print("\nChunks with unknown section titles:")
for chunk in unknown_chunks:
    print(
        f"Page {chunk['page_number']} | "
        f"Length {len(chunk['text'])}"
    )
    print(chunk["text"][:250])
    print("-" * 40)


print("\nReporting section check:")

for chunk in chunks:
    title = chunk["metadata"]["section_title"]
    if "Reporting and Resolution of Violations" in title:
        print("Page:", chunk["page_number"])
        print("Section:", title)
        print("Text:", chunk["text"][:200])
        print("-" * 40)
