from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Delhi is the capital of india",
    "Kolkata is the capital of west Bengal",
    "Paris is the capital of france"
]

vector = embedding.embed_documents(documents)

print(str(vector))