import fitz

with fitz.open("sample_documents/manual.pdf") as pdf:
    page = pdf[1]  # Page 3

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span["text"] for span in spans).strip()

            if text:
                size = max(span["size"] for span in spans)
                bold = any("bold" in span["font"].lower() for span in spans)
                print(f"Size: {size:.1f} | Bold: {bold} | {text}")