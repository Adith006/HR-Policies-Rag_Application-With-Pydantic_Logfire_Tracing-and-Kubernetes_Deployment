
import fitz

with fitz.open("sample_documents/manual.pdf") as pdf:
    for page_number, page in enumerate(pdf[:2], start=1):
        print(f"\n--- PAGE {page_number} ---")

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span["text"] for span in spans).strip()

                if text:
                    size = max(span["size"] for span in spans)
                    print(f"Font size: {size:.1f} | {text}")
