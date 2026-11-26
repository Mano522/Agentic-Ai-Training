from pathlib import Path

from pypdf import PdfReader, PdfWriter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENT_FOLDER = Path(__file__).parent / "documents"


def load_and_split_pdfs():

    documents = []

    pdf_files = sorted(DOCUMENT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found in the documents folder.")
        return []

    # Extract text from each PDF
    for pdf_file in pdf_files:

        print(f"\nReading: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text and text.strip():

                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_file.name,
                            "page": page_number
                        }
                    )
                )

    if not documents:
        print("No extractable text found in the PDF documents.")
        return []

    # Divide extracted text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print("\nPDF processing completed.")
    print("Total pages with extracted text:", len(documents))
    print("Total text chunks generated:", len(chunks))

    return chunks


if __name__ == "__main__":

    chunks = load_and_split_pdfs()

    if chunks:
        print("\nSample extracted chunk:")
        print("-" * 60)
        print(chunks[0].page_content[:1000])
        print("-" * 60)
        print("\nSource information:", chunks[0].metadata)
