
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

text = "Hepatic steatosis is the accumulation of fat in liver cells."

try:
    vector = embeddings.embed_query(text)

    print("Embedding generated successfully.")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])

except Exception as e:
    print("Embedding failed:")
    print(e)
