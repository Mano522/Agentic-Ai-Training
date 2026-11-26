
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from document_processor import load_and_split_pdfs


VECTOR_DB_PATH = "vector_db"
BATCH_SIZE = 10

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def create_vector_store():

    chunks = load_and_split_pdfs()

    if not chunks:
        print("No document chunks available.")
        return None

    vector_store = Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    # Generate unique, repeatable IDs
    ids = []

    for chunk in chunks:
        content = (
            str(chunk.metadata) + chunk.page_content
        )

        chunk_id = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    print("\nGenerating embeddings in batches...")

    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total_chunks)

        batch_chunks = chunks[start:end]
        batch_ids = ids[start:end]

        vector_store.add_documents(
            documents=batch_chunks,
            ids=batch_ids
        )

        print(f"Processed {end} / {total_chunks} chunks")

    print("\nEmbedding generation completed.")

    print(
        "Total records in ChromaDB:",
        vector_store._collection.count()
    )

    return vector_store


def search_documents(query, vector_store, k=3):

    return vector_store.similarity_search(
        query,
        k=k
    )


if __name__ == "__main__":

    store = create_vector_store()

    if store:

        query = input(
            "\nEnter a research topic to search: "
        )

        results = search_documents(query, store)

        print("\nRelevant document passages:")
        print("=" * 60)

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("Source:", result.metadata.get("source"))
            print("Page:", result.metadata.get("page"))
            print("Content:")
            print(result.page_content[:1000])
            print("-" * 60)
