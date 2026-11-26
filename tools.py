
import csv
from pathlib import Path

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# --------------------------------------------------
# 1. CALCULATOR TOOL
# --------------------------------------------------

@tool
def calculator(operation: str, numbers: list[float]) -> str:
    """Perform numerical calculations: sum, average, maximum or minimum."""

    try:
        if not numbers:
            return "Please provide at least one number."

        if operation == "sum":
            result = sum(numbers)
        elif operation == "average":
            result = sum(numbers) / len(numbers)
        elif operation == "maximum":
            result = max(numbers)
        elif operation == "minimum":
            result = min(numbers)
        else:
            return "Unsupported operation."

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


# --------------------------------------------------
# 2. CSV DATASET SEARCH TOOL
# --------------------------------------------------

@tool
def search_dataset(query: str) -> str:
    """Search all columns of the local research CSV dataset using a keyword."""

    file_path = Path(__file__).parent / "data" / "research_data.csv"

    try:
        results = []

        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                searchable_text = " ".join(row.values()).lower()

                if query.lower() in searchable_text:
                    results.append(row)

        if results:
            return str(results)

        return "No matching records found."

    except FileNotFoundError:
        return "Research dataset file was not found."


# --------------------------------------------------
# 3. CSV TOPIC RETRIEVAL TOOL
# --------------------------------------------------

@tool
def retrieve_information(topic: str) -> str:
    """Retrieve the complete dataset record for a specific research topic."""

    file_path = Path(__file__).parent / "data" / "research_data.csv"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["topic"].lower() == topic.lower():
                    return str(row)

        return f"No information available for {topic}."

    except FileNotFoundError:
        return "Research dataset file was not found."


# --------------------------------------------------
# 4. RESEARCH PDF RETRIEVAL TOOL (RAG)
# --------------------------------------------------

@tool
def search_research_documents(query: str) -> str:
    """Search the uploaded research PDFs for relevant passages.
    Use this tool when the user asks questions about uploaded papers,
    research documents, their findings, methods or conclusions.
    """

    try:
        vector_db_path = str(
            Path(__file__).parent / "vector_db"
        )

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        vector_store = Chroma(
            collection_name="research_documents",
            embedding_function=embeddings,
            persist_directory=vector_db_path
        )

        results = vector_store.similarity_search(
            query,
            k=4
        )

        if not results:
            return "No relevant passages found in the research documents."

        formatted_results = []

        for i, result in enumerate(results, start=1):

            source = result.metadata.get("source", "Unknown")
            page = result.metadata.get("page", "Unknown")

            formatted_results.append(
                f"""
                Result {i}
                Source: {source}
                Page: {page}
                Content:
                {result.page_content}
                """
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Document retrieval error: {e}"


# --------------------------------------------------
# REGISTER ALL TOOLS
# --------------------------------------------------

TOOLS = [
    calculator,
    search_dataset,
    retrieve_information,
    search_research_documents
]
