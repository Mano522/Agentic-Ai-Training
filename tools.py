
import csv
from pathlib import Path

from langchain_core.tools import tool


# Tool 1: Calculator
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


# Tool 2: Search the research dataset
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


# Tool 3: Retrieve a specific research topic
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


# Register all available tools
TOOLS = [
    calculator,
    search_dataset,
    retrieve_information
]
