from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import hashlib
from pinecone import Pinecone


OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
EMBEDDINGS = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]


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


def generate_short_id(content: str) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def map_vector_and_text(
    documents: list[any], doc_embeddings: list[list[float]]
) -> list[dict[str, any]]:
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        # Create a data item dictionary
        data_item = {
            "id": doc_id,
            "values": embedding[0],
            "metadata": {"text": doc_text},  # Include the text as metadata
        }

        # Append the data item to the list
        data_with_metadata.append(data_item)

    return data_with_metadata

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(st.secrets["PINECONE_INDEX"])


def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
    try:
        index.upsert(vectors=data_with_metadata)
        return True
    except Exception as e:
        st.write(e)
        return False


def get_query_embeddings(query: str) -> list[float]:
    query_embeddings = EMBEDDINGS.embed_query(query)
    return query_embeddings


def query_pinecone_index(
    query_embeddings: list, top_k: int = 2, include_metadata: bool = True
) -> dict[str, any]:
    query_response = index.query(
        vector=query_embeddings, top_k=top_k, include_metadata=include_metadata
    )
    return query_response
