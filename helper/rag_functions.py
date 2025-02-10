from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st


OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
EMBEDDINGS = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def read_pdf_from_directory(directory: str) -> list[str]:
    # Initialize a PyPDFDirectoryLoader object with the given directory
    file_loader = PyPDFDirectoryLoader(directory)

    # Load PDF documents from the directory
    documents = file_loader.load()

    # Extract only the page content from each document
    page_contents = [doc.page_content for doc in documents]

    return page_contents


def chunk_text_for_list(docs, max_chunk_size = 1000):
    def chunk_text(text, max_chunk_size):
        # Ensure each text ends with a double newline to correctly split paragraphs
        if not text.endswith("\n\n"):
            text += "\n\n"
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        # Iterate over paragraphs and assemble chunks
        for paragraph in paragraphs:
            # Check if adding the current paragraph exceeds the maximum chunk size
            if (
                len(current_chunk) + len(paragraph) + 2 > max_chunk_size
                and current_chunk
            ):
                # If so, add the current chunk to the list and start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Add the current paragraph to the current chunk
            current_chunk += paragraph.strip() + "\n\n"
        # Add any remaining text as the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # Apply the chunk_text function to each document in the list
    return [chunk_text(doc, max_chunk_size) for doc in docs]


def generate_embeddings(documents: list[any]) -> list[list[float]]:
    embedded = [EMBEDDINGS.embed_documents(doc) for doc in documents]
    return embedded
