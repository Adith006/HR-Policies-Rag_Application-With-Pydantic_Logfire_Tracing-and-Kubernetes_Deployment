from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Employees need manager approval for annual leave.",
    "The company provides a remote work policy.",
]

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))
print("Vector dimensions:", embeddings.shape[1])
print("First vector preview:", embeddings[0][:5])